from pathlib import Path

from xai_pipeline.pipelines.run_pipeline import run_pipeline


def test_dry_run_writes_manifest(tmp_path: Path) -> None:
    source = Path("configs/base.yaml").read_text(encoding="utf-8")
    source = source.replace("output_root: outputs", f"output_root: {tmp_path.as_posix()}")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(source, encoding="utf-8")
    manifest = run_pipeline(config_path, dry_run=True)
    assert manifest.exists()
