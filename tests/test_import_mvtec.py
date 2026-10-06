import io
import tarfile
import zipfile
from pathlib import PurePosixPath

import pytest

from scripts.import_mvtec_leather import import_leather, main, relative_in_category
from scripts.make_fake_leather import make_dataset


@pytest.fixture(scope="module")
def mvtec_like(tmp_path_factory):
    """A tiny 'full MVTec' tree: leather plus another category that must be ignored."""
    root = tmp_path_factory.mktemp("src") / "mvtec_anomaly_detection"
    make_dataset(root / "leather", size=64, n_train=2, n_test_good=2, per_defect=1)
    (root / "leather" / "FAKE_DATA.txt").unlink()  # real data has no marker
    (root / "leather" / "_fake_manifest.csv").unlink()
    (root / "bottle" / "train" / "good").mkdir(parents=True)
    (root / "bottle" / "train" / "good" / "000.png").write_bytes(b"not leather")
    (root / "readme.txt").write_text("top-level readme")
    return root


def listing(path):
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())


def test_relative_in_category_rules():
    assert relative_in_category("mvtec_anomaly_detection/leather/test/cut/000.png") == PurePosixPath("test/cut/000.png")
    assert relative_in_category("leather/train/good/001.png") == PurePosixPath("train/good/001.png")
    assert relative_in_category("mvtec_anomaly_detection/bottle/train/good/000.png") is None
    assert relative_in_category("leather/../../etc/passwd.png") is None
    assert relative_in_category("/leather/test/x.png") is None
    assert relative_in_category("leather/test/evil.sh") is None
    assert relative_in_category("leather/") is None


def test_import_from_folder(mvtec_like, tmp_path):
    dest = tmp_path / "data" / "leather"
    n = import_leather(mvtec_like, dest)
    assert n == len(listing(mvtec_like / "leather"))
    assert listing(dest) == listing(mvtec_like / "leather")
    assert not (dest / "bottle").exists()


def test_import_from_tar_xz_full_archive(mvtec_like, tmp_path):
    archive = tmp_path / "mvtec_anomaly_detection.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(mvtec_like, arcname="mvtec_anomaly_detection")
    dest = tmp_path / "out"
    import_leather(archive, dest)
    assert listing(dest) == listing(mvtec_like / "leather")


def test_import_from_per_category_tar(mvtec_like, tmp_path):
    archive = tmp_path / "leather.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(mvtec_like / "leather", arcname="leather")
    dest = tmp_path / "out"
    import_leather(archive, dest)
    assert listing(dest) == listing(mvtec_like / "leather")


def test_import_from_zip(mvtec_like, tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in mvtec_like.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(mvtec_like.parent).as_posix())
    dest = tmp_path / "out"
    import_leather(archive, dest)
    assert listing(dest) == listing(mvtec_like / "leather")


def test_path_traversal_members_are_ignored(tmp_path):
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        for name, data in [("leather/test/cut/000.png", b"ok"), ("leather/../../escaped.png", b"bad")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    dest = tmp_path / "work" / "out"
    import_leather(archive, dest)
    assert listing(dest) == ["test/cut/000.png"]
    assert not (tmp_path / "escaped.png").exists()


def test_errors(mvtec_like, tmp_path):
    with pytest.raises(FileNotFoundError):
        import_leather(tmp_path / "nope.zip", tmp_path / "o")
    other = tmp_path / "other.tar"
    with tarfile.open(other, "w") as tar:
        info = tarfile.TarInfo("bottle/a.png")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="No 'leather'"):
        import_leather(other, tmp_path / "o2")
    assert not (tmp_path / "o2").exists()  # nothing left behind
    with pytest.raises(ValueError, match="Don't know"):
        bad = tmp_path / "x.rar"
        bad.write_bytes(b"x")
        import_leather(bad, tmp_path / "o3")
    dest = tmp_path / "full"
    import_leather(mvtec_like, dest)
    with pytest.raises(FileExistsError):
        import_leather(mvtec_like, dest)
    import_leather(mvtec_like, dest, force=True)


def test_main_runs_the_setup_check(mvtec_like, tmp_path, capsys):
    code = main([str(mvtec_like), "--out", str(tmp_path / "leather")])
    out = capsys.readouterr().out
    assert code == 0 and "data kind: looks like real data" in out and "masks: every defect image has a mask" in out
