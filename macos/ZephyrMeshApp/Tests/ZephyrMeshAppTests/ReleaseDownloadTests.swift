import XCTest
import CryptoKit
@testable import ZephyrMeshApp

/// Serves canned responses so the streaming download can be tested without a network.
final class StubDownloadProtocol: URLProtocol {
    struct Reply { var status = 200; var body = Data(); var declaredLength: Int? = nil }
    nonisolated(unsafe) static var reply = Reply()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let reply = Self.reply
        var headers = ["Content-Type": "application/zip"]
        if let length = reply.declaredLength { headers["Content-Length"] = String(length) }
        let response = HTTPURLResponse(url: request.url!, statusCode: reply.status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        // Several chunks, like a real transfer.
        var offset = 0
        while offset < reply.body.count {
            let end = min(offset + 64 * 1024, reply.body.count)
            client?.urlProtocol(self, didLoad: reply.body.subdata(in: offset..<end))
            offset = end
        }
        client?.urlProtocolDidFinishLoading(self)
    }
}

final class ReleaseDownloadTests: XCTestCase {
    private var session: URLSession!
    private var directory: URL!

    override func setUpWithError() throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubDownloadProtocol.self]
        session = URLSession(configuration: configuration)
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("zephyr-download-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        session.invalidateAndCancel()
        try? FileManager.default.removeItem(at: directory)
    }

    private var request: URLRequest { URLRequest(url: URL(string: "https://github.com/Whyater/Zephyr-Mesh/releases/download/v9.9.9/x.zip")!) }

    func testStreamsToDiskAndReturnsTheSHA256OfWhatArrived() async throws {
        let body = Data((0..<(3 * 1_048_576 + 17)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        StubDownloadProtocol.reply = .init(body: body)
        let destination = directory.appendingPathComponent("x.zip")
        let digest = try await ReleaseUpdater.download(request, to: destination, maximumBytes: 8 * 1_048_576, session: session)
        XCTAssertEqual(digest, SHA256.hash(data: body).map { String(format: "%02x", $0) }.joined())
        XCTAssertEqual(try Data(contentsOf: destination), body)
    }

    func testStopsWhenTheBodyIsLargerThanTheLimitAndDeletesThePartialFile() async throws {
        StubDownloadProtocol.reply = .init(body: Data(count: 3 * 1_048_576))
        let destination = directory.appendingPathComponent("x.zip")
        do {
            _ = try await ReleaseUpdater.download(request, to: destination, maximumBytes: 1_048_576, session: session)
            XCTFail("an oversized download must be refused")
        } catch ReleaseUpdaterError.archiveTooLarge(let limit) {
            XCTAssertEqual(limit, 1_048_576)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    }

    func testRefusesADeclaredLengthOverTheLimitBeforeWritingAnything() async throws {
        StubDownloadProtocol.reply = .init(body: Data(count: 1024), declaredLength: 1024)
        let destination = directory.appendingPathComponent("x.zip")
        do {
            _ = try await ReleaseUpdater.download(request, to: destination, maximumBytes: 512, session: session)
            XCTFail("a declared oversized download must be refused")
        } catch ReleaseUpdaterError.archiveTooLarge {}
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    }

    func testHTTPErrorsBecomeAReadableError() async throws {
        StubDownloadProtocol.reply = .init(status: 404, body: Data("not found".utf8))
        let destination = directory.appendingPathComponent("x.zip")
        do {
            _ = try await ReleaseUpdater.download(request, to: destination, session: session)
            XCTFail("HTTP 404 must fail")
        } catch ReleaseUpdaterError.httpStatus(let status) {
            XCTAssertEqual(status, 404)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    }

    func testStreamedDigestIsCheckedAgainstTheRelease() throws {
        let asset = ZephyrReleaseAsset(name: "x.zip", browserDownloadURL: "https://github.com/Whyater/Zephyr-Mesh/releases/download/v9.9.9/x.zip", digest: "sha256:" + String(repeating: "A", count: 64))
        XCTAssertEqual(try ReleaseUpdater.verify(sha256: String(repeating: "a", count: 64), against: asset), String(repeating: "a", count: 64))
        XCTAssertThrowsError(try ReleaseUpdater.verify(sha256: String(repeating: "b", count: 64), against: asset))
    }
}
