"""Wire-protocol framing for the committee bridge."""
import json

import pytest

from mliprun.core.committee import protocol


class TestEncodeDecode:
    def test_round_trip_preserves_a_calc_request(self):
        request = protocol.calc_request(
            numbers=[29, 29],
            positions=[[0.0, 0.0, 0.0], [1.8, 1.8, 0.0]],
            cell=[[3.6, 0.0, 0.0], [0.0, 3.6, 0.0], [0.0, 0.0, 3.6]],
            pbc=[True, True, True],
        )
        assert protocol.decode(protocol.encode(request)) == request

    def test_encoded_message_is_exactly_one_line(self):
        blob = protocol.encode(protocol.calc_request([29], [[0.0, 0.0, 0.0]],
                                                     [[1.0, 0.0, 0.0]] * 3,
                                                     [True, True, True]))
        assert blob.endswith(b"\n")
        assert blob.count(b"\n") == 1

    def test_encode_rejects_nan(self):
        """A NaN energy must fail loudly in the member's own process rather
        than poison the committee mean."""
        with pytest.raises(protocol.ProtocolError):
            protocol.encode(protocol.ok(energy=float("nan")))

    def test_encode_rejects_infinity(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.encode(protocol.ok(energy=float("inf")))

    def test_encode_rejects_unserializable_value(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.encode({"cmd": "calc", "positions": object()})

    def test_decode_accepts_str_and_bytes_identically(self):
        text = json.dumps({"ok": True, "energy": -1.5})
        assert protocol.decode(text) == protocol.decode(text.encode("utf-8"))

    def test_decode_rejects_malformed_json(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(b'{"ok": true')

    def test_decode_rejects_an_empty_line(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(b"\n")

    def test_decode_rejects_a_json_array(self):
        """A bare list is valid JSON but not a message."""
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(b"[1, 2, 3]")

    def test_decode_rejects_invalid_utf8(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(b"\xff\xfe\n")


class TestMessages:
    def test_load_request_carries_every_selector(self):
        request = protocol.load_request("uma-s-1p2", uma_task="oc20",
                                        device="cuda")
        assert request["cmd"] == "load"
        assert request["protocol"] == protocol.PROTOCOL_VERSION
        assert request["mlip"] == "uma-s-1p2"
        assert request["uma_task"] == "oc20"
        assert request["mace_head"] is None
        assert request["sevennet_task"] is None
        assert request["device"] == "cuda"

    def test_calc_request_coerces_sequences_to_lists(self):
        request = protocol.calc_request((29, 29), [[0.0] * 3] * 2,
                                        [[1.0, 0.0, 0.0]] * 3, (True,) * 3)
        assert request["numbers"] == [29, 29]
        assert request["pbc"] == [True, True, True]

    def test_ok_and_error_are_distinguishable(self):
        assert protocol.ok(energy=-1.0)["ok"] is True
        failed = protocol.error(ValueError("boom"), "Traceback ...")
        assert failed["ok"] is False
        assert failed["error"] == "boom"
        assert failed["traceback"] == "Traceback ..."

    def test_error_without_traceback_is_an_empty_string(self):
        assert protocol.error("boom")["traceback"] == ""
