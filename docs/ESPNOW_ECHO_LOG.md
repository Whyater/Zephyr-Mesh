# ESP-NOW echo log format (proposal)

Status: proposal on a branch. The bench firmware that would write this format
is not written yet, and every log in the repo is synthetic. The format uses as
few columns as possible so that firmware is easy to write.

This page covers the plain text log two ESP32 boards write during an echo
test. `tools/import_espnow_log.py` turns the log into a `zephyr-espnow-trace-1`
trace, and `tools/fit_espnow_log.py` turns that trace into simulator link
settings.

---

## The test

Board A (the sender) sends a numbered packet at a fixed rate. Board B (the
echoer) sends each packet straight back. Both boards print one line per
packet to their own USB serial port, and the laptop saves each port to a file.

- A stamps the send time and the echo arrival time with its own clock.
- B stamps the arrival time and the echo send time with its own clock.
- Each board starts its clock at boot, so the clocks have different origins
  and a time from A can never be subtracted from a time from B.

### Limits of echo timing

The round trip is a real measurement, because each piece uses one clock:

```text
rtt = (echo_us - tx_us)            on A's clock
    - (echo_tx_us - rx_us)         on B's clock (how long B held the packet)
```

The one-way delay is then **estimated** as `rtt / 2`. That only works if the
trip from A to B and the trip from B to A take the same time on average. If
the A to B trip really takes `d_out` and the way back takes `d_back`, then
`rtt / 2 = (d_out + d_back) / 2`, which is off from `d_out` by
`(d_back - d_out) / 2`. Nothing in an echo log can detect that difference. The
only way to check it is a shared clock signal, such as a wire from a GPIO pin
on one board to the other.

The spread is also different. If the two trips vary independently by about
the same amount, `rtt / 2` varies less than one trip does (by a factor of
about the square root of 2). The fit tool corrects for that and says so in
its output.

---

## File 1: sender log (board A, required)

```text
# zephyr-espnow-echo-log-1
# role=sender
# board=A
# peer=B
# session=a-boot-0001
# firmware=echo-bench-0.1
# payload_bytes=32
# channel=1
# rate_hz=50
seq,tx_us,echo_us
0,1000123,1003701
1,1020110,
2,1040098,1043655
```

| Column | Meaning |
|---|---|
| `seq` | Packet number. Starts anywhere, goes up by exactly 1 each row. Use a 32-bit counter so it never wraps during a test. |
| `tx_us` | A's clock when A handed the packet to ESP-NOW, in whole microseconds (`esp_timer_get_time()`). |
| `echo_us` | A's clock when the echo came back. Leave it empty if no echo came back. If two echoes come back for one packet, log the first. |

## File 2: echoer log (board B, optional but strongly recommended)

```text
# zephyr-espnow-echo-log-1
# role=echoer
# board=B
# peer=A
# session=b-boot-0001
seq,rx_us,echo_tx_us,rssi_dbm
0,553001,553042,-47
2,592977,593015,-48
```

| Column | Meaning |
|---|---|
| `seq` | The packet number B received, copied from the packet. One row per arrival, in arrival order. |
| `rx_us` | B's clock when the packet arrived. |
| `echo_tx_us` | B's clock when B sent the echo. Empty if B did not echo it. |
| `rssi_dbm` | Signal strength B saw for this packet, a whole number in dBm such as `-47`. Empty if not available. |

Without the echoer log, a missing echo cannot be blamed on either direction,
so every packet without an echo becomes `unknown` and the loss fit has nothing
to work with. The round trip then also includes B's hold time.

---

## Rules (the reader rejects anything else)

The reader stops at the first problem and names the file and line number.
It never skips or repairs a line.

1. Text is UTF-8. Lines end in `\n` (a `\r\n` ending is accepted and the
   `\r` is dropped).
2. Line 1 is exactly `# zephyr-espnow-echo-log-1`.
3. Then `# key=value` lines with one key per line, written with `=` directly
   between key and value. Each key appears at most once.
   - Both files need `role`, `board`, `peer`, `session`.
   - The sender file also needs `firmware`.
   - Optional in either file: `payload_bytes` (whole number), `channel`
     (1 to 14), `rate_hz` (positive number), `phy`, `note`.
   - Any other key is an error.
4. Then the column header, exactly `seq,tx_us,echo_us` (sender) or
   `seq,rx_us,echo_tx_us,rssi_dbm` (echoer).
5. Then data rows only, each with exactly the header's number of fields.
   Blank lines, comments, spaces and board boot messages are all errors.
6. `seq` and every time are whole numbers of zero or more, written as plain
   digits. `rssi_dbm` is a whole number from -128 to 0.
7. Sender file: `seq` goes up by exactly 1 each row. A jump means serial lines
   were lost, which is an error, not packet loss. `tx_us` always goes up.
   `echo_us` is never earlier than `tx_us`.
8. Echoer file: `rx_us` never goes down. `echo_tx_us` is never earlier than
   `rx_us`. A `seq` may repeat (a duplicate arrival) or arrive out of order.
   Every `seq` must also appear in the sender file.
9. The echoer's `peer` is the sender's `board`, and the sender's `peer` is
   the echoer's `board`.
10. If A logged an echo for a packet, B must have logged that packet with an
    `echo_tx_us`. Otherwise one of the two serial logs is incomplete, and the
    reader refuses to guess which.

## Packet labels in the trace

One trace record per sender row, with `clock_domain = "sender_round_trip"`:

| Situation | `outcome` |
|---|---|
| A got an echo | `received` |
| No echo, B logged the packet | `received` (the echo was lost on the way back) |
| No echo, B never logged it, and its `seq` is inside the range B logged | `lost` (`loss_reason = not_in_echoer_log`) |
| No echo, B never logged it, and its `seq` is before B's first or after B's last logged `seq` | `unknown` (B's log may not cover it) |
| No echo and no echoer log at all | `unknown` |

A repeated arrival in B's log becomes an extra record marked
`duplicate = true`. Duplicates are left out of the loss fit.

Times in the trace are seconds: `send_time_s = tx_us / 1e6`,
`echo_time_s = echo_us / 1e6`, `receive_time_s = rx_us / 1e6` (B's clock),
`echo_turnaround_s = (echo_tx_us - rx_us) / 1e6`.

---

## Commands

```bash
python tools/import_espnow_log.py --sender a.csv --echoer b.csv \
    --status synthetic --trace-id bench-1 --out trace.json
python tools/fit_espnow_log.py trace.json --out fit.json
```

`--status measured` also needs the capture details (operator, capture time,
code revision, source). The reader records the SHA-256 of both raw files.

`tests/data/espnow_echo/` holds a synthetic example made from known settings.
It is not a measurement.
