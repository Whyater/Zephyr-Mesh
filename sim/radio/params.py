"""Sourced parameters and presets for the radio layer.

Every number here comes from outside the project and is **literature-derived**
or taken from a standard; none is measured on Zephyr hardware.  Each entry
carries a plausible range, the source, and the date it was checked, so
experiments can sweep the range wherever it matters.

Status meanings:

- ``standard``: a formula or constant from a standard (or open code that
  implements it, when the standard text itself was not opened).
- ``literature-derived``: a published value, used as a default, with a range.
- ``partial``: one weaker source (a GitHub issue, a forum, a survey table) or
  a value from a nearby setting (another band, an office instead of a gym).
  Used only where a model needs some number, and always swept.
- ``UNCONFIRMED``: no source found.  Kept out of defaults.
- ``modelling-choice``: not a fact about the world; a choice of this model.

The sources were checked on 2026-10-14.
"""
from __future__ import annotations

from dataclasses import dataclass

from .channel import ChannelConfig

CHECKED = "2026-10-14"


@dataclass(frozen=True)
class SourcedValue:
    value: float | None
    low: float | None
    high: float | None
    units: str
    source: str
    checked: str
    status: str
    note: str = ""


_ESPNOW_GUIDE = "https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html"
_ESP32_DATASHEET = "https://documentation.espressif.com/esp32_datasheet_en.pdf"
_ESP32S3_DATASHEET = "https://documentation.espressif.com/esp32-s3_datasheet_en.pdf"
_LINUX_MAC80211 = "https://raw.githubusercontent.com/torvalds/linux/master/net/mac80211/util.c"
_FREEBSD_PHY = "https://cgit.freebsd.org/src/plain/sys/net80211/ieee80211_phy.h"
_HU_BERLIN = "https://sarwiki.informatik.hu-berlin.de/Packet_transmission_time_in_802.11"
_ITU_P525 = "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.525-3-201611-S!!PDF-E.pdf"
_ITU_P1238 = "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1238-9-201706-I!!PDF-E.pdf"
_3GPP_38901 = "https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.04.00_60/tr_138901v190400p.pdf"
_KHUWAJA = "https://wrap.warwick.ac.uk/id/eprint/104438/1/WRAP-survey-channel-modeling-UAV-communications-Chen-2018.pdf"
_YANMAZ = "https://data.bettstetter.com/publications/yanmaz-2013-infocom-drones.pdf"
_BALDANSA = "https://arxiv.org/pdf/2211.10699"
_JANUSZKIEWICZ = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6211019/"
_WILSON_USC = "https://www.am1.us/wp-content/uploads/Documents/E10589_Propagation_Losses_2_and_5GHz.pdf"
_HUM_POLARIZATION = "https://www.waves.utoronto.ca/prof/svhum/ece422/notes/12-friiseffects.pdf"
_GUDMUNDSON = "https://span.ece.utah.edu/uploads/gudmundson1991correlation-shadow-fading.pdf"
_DASILVA_GE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC9696961/"
_IDF_ISSUE_9383 = "https://github.com/espressif/esp-idf/issues/9383"
_HACKADAY_ESPNOW = "https://hackaday.io/project/161896-linux-espnow/log/154698-how-esp-now-really-works"
_NTU_BLOG = "https://blogs.ntu.edu.sg/ps9888-2020-g16/author/jong114/page/2"
_JAIIO_2026 = "https://55jaiio.sadio.org.ar/wp-content/uploads/2026/07/273.pdf"
_JOHNSON = "https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise"
_ESP32_FORUM_NF = "https://www.esp32.com/viewtopic.php?t=3096"


def _v(value, low, high, units, source, status, note=""):
    return SourcedValue(value, low, high, units, source, CHECKED, status, note)


PARAMS: dict[str, SourcedValue] = {
    # ---- frame format and PHY (ESP-NOW v1 over 802.11b DSSS) ----
    "espnow_overhead_bytes": _v(43, 43, 43, "bytes", _ESPNOW_GUIDE, "literature-derived",
                                "24 MAC header + 1 category + 3 OUI + 4 random + 7 vendor element header + 4 FCS"),
    "espnow_max_payload_bytes": _v(250, 250, 1470, "bytes", _ESPNOW_GUIDE, "literature-derived",
                                   "250 for ESP-NOW v1, 1470 for v2"),
    "espnow_default_rate_mbps": _v(1.0, 1.0, 54.0, "Mbps", _ESPNOW_GUIDE, "literature-derived",
                                   "default 1 Mbps long preamble; per-peer rate configurable"),
    "dsss_long_plcp_s": _v(192e-6, 192e-6, 192e-6, "s", _LINUX_MAC80211, "standard",
                           "144 us preamble + 48 us PLCP header; 1 Mbps always uses the long form"),
    "dsss_short_plcp_s": _v(96e-6, 96e-6, 96e-6, "s", _LINUX_MAC80211, "standard", "2, 5.5 and 11 Mbps only"),
    "slot_s": _v(20e-6, 9e-6, 20e-6, "s", _FREEBSD_PHY, "standard",
                 "20 us for DSSS/CCK; 9 us short slot for ERP. Which one ESP32 uses for ESP-NOW is UNCONFIRMED"),
    "sifs_s": _v(10e-6, 10e-6, 10e-6, "s", _FREEBSD_PHY, "standard", "2.4 GHz"),
    "cw_min": _v(31, 15, 31, "slots", _FREEBSD_PHY, "standard",
                 "31 for DSSS/CCK, 15 for OFDM; ESP32 value for ESP-NOW UNCONFIRMED"),
    "cw_max": _v(1023, 1023, 1023, "slots", _FREEBSD_PHY, "standard"),
    "ack_bytes": _v(14, 14, 14, "bytes", _HU_BERLIN, "standard", "sent at a basic rate, 1 Mbps for 1 Mbps data"),
    # ---- ACK and retries (the prompt's "do not assume" item) ----
    "unicast_retry_limit": _v(5, 0, 10, "retransmissions", _IDF_ISSUE_9383, "partial",
                              "Espressif documents that unicast success means 'received successfully on the MAC "
                              "layer' and offers no retry setting. The count is reported only in a GitHub issue "
                              "('up to 5'); an ESP8266 sniffer log saw 11 attempts with rate fallback "
                              f"({_HACKADAY_ESPNOW}). Always swept 0, 5, 10."),
    "broadcast_acked": _v(0, 0, 0, "bool", _NTU_BLOG, "partial",
                          "broadcast gets no ACK and no retries by sniffer observation and 802.11 rules; "
                          "Espressif docs are silent"),
    # ---- radio front end (ESP32 datasheet) ----
    "tx_power_dbm": _v(19.5, 2.0, 20.5, "dBm", _ESP32_DATASHEET, "literature-derived",
                       "11b typical 19.5 (18.5 to 20.5); software range 2 to 20 dBm"),
    "sensitivity_1mbps_dbm": _v(-98.0, -98.0, -98.0, "dBm", _ESP32_DATASHEET, "literature-derived",
                                "11b 1 Mbps typical"),
    "sensitivity_per": _v(0.08, 0.08, 0.08, "fraction", _ESP32S3_DATASHEET, "literature-derived",
                          "8 % PER for 11b (ESP32-S3 datasheet; the classic ESP32 datasheet does not state it)"),
    "sensitivity_ref_bytes": _v(1024, 1024, 1024, "bytes", "https://warpproject.org/trac/wiki/802.11/Benchmarks/Rx_Char",
                                "standard", "802.11 DSSS sensitivity test length"),
    "thermal_noise_20mhz_dbm": _v(-101.0, -101.0, -101.0, "dBm", _JOHNSON, "standard", "-174 dBm/Hz + 10 log10(20e6)"),
    "noise_figure_db": _v(4.5, 3.0, 6.0, "dB", _ESP32_FORUM_NF, "partial",
                          "Espressif staff forum post seen only as a search excerpt"),
    "noise_floor_dbm": _v(-96.5, -98.0, -90.0, "dBm", _ESP32_FORUM_NF, "partial",
                          "thermal + noise figure; a measured ESP32 floor is UNCONFIRMED. Only the gap between the "
                          "actual floor and the reference floor changes PER, so the default sets them equal"),
    "antenna_gain_dbi": _v(None, None, None, "dBi", "", "UNCONFIRMED",
                           "ESP32 PCB antenna pattern not found; 0 dBi assumed and orientation loss swept instead"),
    # ---- propagation ----
    "pl0_1m_db": _v(40.2, 39.6, 40.3, "dB", _ITU_P525, "standard", "free space at 2.44 GHz, 1 m; 40.05 at 2.40 GHz"),
    "exponent_indoor_large_room": _v(2.0, 1.7, 2.2, "-", _ITU_P1238, "partial",
                                     f"ITU large open rooms ~2.0; 3GPP InH LOS 1.73, InF LOS 2.15 ({_3GPP_38901}); "
                                     "no measured gym at 2.4 GHz"),
    "sigma_indoor_large_room_db": _v(3.0, 2.3, 4.3, "dB", _3GPP_38901, "partial", "3GPP InH LOS 3, InF LOS 4.3"),
    "exponent_outdoor_open": _v(2.0, 2.0, 2.6, "-", _KHUWAJA, "partial",
                                f"Yanmaz 2.01 at 5.24 GHz ({_YANMAZ}); 2.3 and 2.6 for 2.4 GHz air-to-air rows in the survey"),
    "sigma_outdoor_open_db": _v(2.5, 2.5, 4.0, "dB", _YANMAZ, "partial", "measured at 5 GHz; UNCONFIRMED at 2.4 GHz"),
    "shadowing_corr_distance_m": _v(10.0, 6.0, 50.0, "m", _3GPP_38901, "partial",
                                    f"3GPP InH LOS 10, NLOS 6; Gudmundson microcell ~8 m ({_GUDMUNDSON})"),
    "ground_breakpoint_note": _v(None, None, None, "m", _BALDANSA, "partial",
                                 "dual-slope breakpoint 4 h_t h_r / wavelength; ESP32 data sit ~10 dB above the ideal "
                                 "n = 4 curve past 20 m, so n_far is swept 2 to 4 (modelling choice)"),
    "body_loss_db": _v(10.0, 10.0, 21.0, "dB", _JANUSZKIEWICZ, "literature-derived", "up to 25 dB worst case"),
    "wall_drywall_db": _v(0.5, 0.5, 3.0, "dB", _WILSON_USC, "partial", "measured at 2.3 GHz; '3 dB' rule of thumb UNCONFIRMED"),
    "wall_brick_db": _v(4.4, 4.0, 6.0, "dB", _WILSON_USC, "partial", "red brick at 2.3 GHz"),
    "orientation_loss_db": _v(0.0, 0.0, 6.0, "dB", _HUM_POLARIZATION, "literature-derived",
                              "polarization mismatch cos^2: 3 dB at 45 deg, 6 dB at 60 deg; rare nulls deeper"),
    "gilbert_elliott_example": _v(None, None, None, "-", _DASILVA_GE, "partial",
                                  "campus 802.11n fit p = 0.0393, r = 0.1862 (mean bad burst 5.4); the paper says GE "
                                  "misses heavy-tailed bursts. Not ESP-NOW data; swept, never defaulted"),
    # ---- stack and model choices ----
    "processing_delay_s": _v(0.0, 0.0, 1.5e-3, "s", _JAIIO_2026, "modelling-choice",
                             "default 0 (unverified). A 2026 paper's ~2.4 ms one-way ESP-NOW estimate minus "
                             "~0.84 ms channel time suggests up to ~1.5 ms of stack time (partial); swept"),
    "queue_capacity_frames": _v(16, 4, 64, "frames", "", "modelling-choice", "ESP-NOW internal queue depth UNCONFIRMED"),
    "cca_threshold_dbm": _v(-82.0, -95.0, -62.0, "dBm", "", "modelling-choice",
                            "-82 dBm is the 802.11 OFDM preamble-detect level; the ESP32 DSSS carrier-sense level is "
                            "UNCONFIRMED. Inside a gym every node is far above any of these"),
    "measured_throughput_250b_pps": _v(357, None, None, "packets/s", _NTU_BLOG, "partial",
                                       "1 Mbps unicast, 250 B: a check value for the airtime model (predicts ~312 to 345)"),
}


def channel_preset(name: str, **overrides) -> ChannelConfig:
    """Literature-derived channel presets.  Ranges live in PARAMS; sweep them.

    - ``indoor_gym``: one large open room, line of sight, n = 2.0, sigma 3 dB.
    - ``outdoor_open``: open field, line of sight, n = 2.0, sigma 2.5 dB.
      Ground bounce is off by default; pass ``breakpoint_m`` and
      ``exponent_far`` to turn on the dual-slope option.
    """
    p = PARAMS
    common = dict(
        tx_power_dbm=p["tx_power_dbm"].value,
        pl0_db=p["pl0_1m_db"].value,
        d0_m=1.0,
        noise_floor_dbm=p["noise_floor_dbm"].value,
        reference_noise_dbm=p["noise_floor_dbm"].value,
        sensitivity_dbm=p["sensitivity_1mbps_dbm"].value,
        sensitivity_per=p["sensitivity_per"].value,
        sensitivity_ref_bytes=int(p["sensitivity_ref_bytes"].value),
        per_model="dbpsk",
        body_loss_db=p["body_loss_db"].value,
        wall_loss_db=p["wall_drywall_db"].value,
        orientation_loss_db=p["orientation_loss_db"].value,
    )
    if name == "indoor_gym":
        cfg = dict(common, exponent=p["exponent_indoor_large_room"].value,
                   shadowing_sigma_db=p["sigma_indoor_large_room_db"].value)
    elif name == "outdoor_open":
        cfg = dict(common, exponent=p["exponent_outdoor_open"].value,
                   shadowing_sigma_db=p["sigma_outdoor_open_db"].value)
    else:
        raise ValueError("preset must be indoor_gym or outdoor_open")
    cfg.update(overrides)
    return ChannelConfig(**cfg)


PRESET_NAMES = ("indoor_gym", "outdoor_open")


def default_phy(**overrides):
    """802.11b DSSS 1 Mbps long preamble, the ESP-NOW default rate."""
    from .mac import PhyConfig

    p = PARAMS
    cfg = dict(rate_mbps=1.0, modulation="dsss", preamble="long", slot_s=p["slot_s"].value, sifs_s=p["sifs_s"].value,
               cw_min=int(p["cw_min"].value), cw_max=int(p["cw_max"].value),
               mac_overhead_bytes=int(p["espnow_overhead_bytes"].value), ack_bytes=int(p["ack_bytes"].value),
               ack_rate_mbps=1.0, max_payload_bytes=int(p["espnow_max_payload_bytes"].value))
    cfg.update(overrides)
    return PhyConfig(**cfg)


def default_mac(**overrides):
    """Carrier sense with backoff; the retry limit is a partial-source value that experiments sweep."""
    from .mac import MacConfig

    p = PARAMS
    cfg = dict(carrier_sense=True, backoff=True, unicast_retry_limit=int(p["unicast_retry_limit"].value),
               queue_capacity=int(p["queue_capacity_frames"].value),
               processing_delay_s=p["processing_delay_s"].value, cca_threshold_dbm=p["cca_threshold_dbm"].value)
    cfg.update(overrides)
    return MacConfig(**cfg)


def as_table() -> list[dict]:
    """Rows for experiment notes and manifests."""
    return [dict(name=k, value=v.value, low=v.low, high=v.high, units=v.units, status=v.status,
                 source=v.source, checked=v.checked, note=v.note) for k, v in sorted(PARAMS.items())]
