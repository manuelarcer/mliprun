"""POST/GET /api/uploads: ASE parsing, multi-frame, failures, limits."""
import json

import numpy as np
import pytest
from ase.build import bulk, molecule


def test_poscar_without_extension_is_detected_as_vasp(client, upload_atoms, runs_dir):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    info = upload_atoms(client, atoms, "POSCAR", fmt="vasp")
    assert info["ok"] is True
    assert info["format"] == "vasp"
    assert info["filename"] == "POSCAR"
    assert info["natoms"] == 4 and info["n_frames"] == 1
    assert info["formula"] == "Cu4"
    assert info["symbols"] == ["Cu"] * 4
    assert info["pbc"] == [True, True, True]
    assert np.allclose(info["cell"], atoms.cell.array)
    assert info["has_constraints"] is False
    assert info["frames_xyz"] is None
    # xyz: natoms line, extended-XYZ comment line, one line per atom
    lines = info["xyz"].strip().splitlines()
    assert lines[0] == "4"
    assert lines[1].startswith('Lattice="3.60000000 0.00000000')
    assert 'Properties=species:S:1:pos:R:3 pbc="T T T"' in lines[1]
    assert len(lines) == 6
    sym, *xyz = lines[3].split()
    assert sym == "Cu" and np.allclose([float(v) for v in xyz], atoms.positions[1], atol=1e-6)
    # stored on disk with meta.json (no xyz inside)
    d = runs_dir / "_uploads" / info["id"]
    assert (d / "POSCAR").is_file()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["format"] == "vasp" and "xyz" not in meta and "frames_xyz" not in meta


def test_poscar_autodetect_without_format_part(client, H, tmp_path):
    import ase.io

    p = tmp_path / "POSCAR"
    ase.io.write(str(p), bulk("Cu", "fcc", a=3.6), format="vasp")
    r = client.post("/api/uploads", headers=H, files=[("files", ("POSCAR", p.read_bytes()))])
    assert r.status_code == 200
    (info,) = r.json()["uploads"]
    assert info["ok"] and info["format"] == "vasp" and info["natoms"] == 1


def test_extxyz_multiframe_describes_last_frame(client, upload_atoms):
    frames = []
    for i in range(3):
        a = bulk("Cu", "fcc", a=3.6, cubic=True)
        a.positions[0, 0] += 0.1 * i
        frames.append(a)
    frames[-1] = frames[-1].repeat((2, 1, 1))     # last frame differs: 8 atoms
    info = upload_atoms(client, frames, "band.extxyz", fmt="extxyz")
    assert info["ok"] and info["format"] == "extxyz"
    assert info["n_frames"] == 3
    assert info["natoms"] == 8 and info["formula"] == "Cu8"
    blocks = info["frames_xyz"].splitlines()
    assert blocks[0] == "4" and blocks[6] == "4" and blocks[12] == "8"
    # x of atom 0 in frame 1 = 0.1
    assert float(blocks[8].split()[1]) == pytest.approx(0.1, abs=1e-6)


def test_garbage_file_is_a_per_file_error_and_not_stored(client, H, runs_dir):
    r = client.post("/api/uploads", headers=H,
                    files=[("files", ("junk.foo", b"\x00\x01 this is not a structure\n")),
                           ("files", ("ok.xyz", b"1\n\nH 0 0 0\n"))])
    assert r.status_code == 200
    bad, good = r.json()["uploads"]
    assert bad["ok"] is False and bad["id"] is None and bad["error"]
    assert bad["filename"] == "junk.foo"
    assert good["ok"] is True and good["natoms"] == 1
    stored = [p.name for p in (runs_dir / "_uploads").iterdir()]
    assert stored == [good["id"]]


def test_filename_is_sanitized(client, H):
    r = client.post("/api/uploads", headers=H,
                    files=[("files", ("../../evil name!.xyz", b"1\n\nH 0 0 0\n"))])
    (info,) = r.json()["uploads"]
    assert info["ok"] and info["filename"] == "evil_name_.xyz"


def test_no_files_is_400(client, H):
    r = client.post("/api/uploads", headers=H, data={"format": "vasp"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"
    r = client.post("/api/uploads", headers=H, content=b"not multipart",
                    )
    assert r.status_code == 400


def test_too_many_files_is_413(client, H):
    files = [("files", (f"h{i}.xyz", b"1\n\nH 0 0 0\n")) for i in range(51)]
    r = client.post("/api/uploads", headers=H, files=files)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "too_large"
    assert r.json()["error"]["details"] == []


def test_list_and_get_uploads(client, upload_atoms):
    a = upload_atoms(client, molecule("H2O"), "water.xyz")
    b = upload_atoms(client, bulk("Cu"), "cu.cif", fmt="cif")
    lst = client.get("/api/uploads").json()["uploads"]
    assert [u["id"] for u in lst] == [b["id"], a["id"]]          # newest first
    assert all("xyz" not in u and "frames_xyz" not in u for u in lst)
    full = client.get(f"/api/uploads/{a['id']}").json()
    assert full["xyz"] == a["xyz"] and full["formula"] == "H2O"
    assert client.get("/api/uploads/ffffffffffff").status_code == 404
    assert client.get("/api/uploads/not-an-id").json()["error"]["code"] == "not_found"


def test_uploads_survive_restart(make_app, upload_atoms):
    from fastapi.testclient import TestClient

    with TestClient(make_app()) as c:
        info = upload_atoms(c, bulk("Cu"), "POSCAR", fmt="vasp")
    with TestClient(make_app()) as c:
        full = c.get(f"/api/uploads/{info['id']}").json()
        assert full["xyz"] == info["xyz"]      # regenerated from the stored file
        assert c.get("/api/uploads").json()["uploads"][0]["id"] == info["id"]
