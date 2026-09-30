import yaml

from prepare_yaml import prepare_config


def test_no_config_file_returns_without_writing_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_FILE", raising=False)

    prepare_config()

    assert not (tmp_path / "acc_cfg.yml").exists()


def test_patches_rank_host_and_port_from_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_config = tmp_path / "base.yml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "machine_rank": 0,
                "main_process_ip": "127.0.0.1",
                "main_process_port": 1234,
                "num_machines": 2,
            }
        )
    )

    monkeypatch.setenv("CONFIG_FILE", str(base_config))
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.5")
    monkeypatch.setenv("MASTER_PORT", "6000")

    prepare_config()

    output = yaml.safe_load((tmp_path / "acc_cfg.yml").read_text())
    assert output["machine_rank"] == 3
    assert output["main_process_ip"] == "10.0.0.5"
    assert output["main_process_port"] == 6000
    assert output["num_machines"] == 2  # untouched keys are passed through


def test_missing_optional_env_vars_leaves_config_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_config = tmp_path / "base.yml"
    base_config.write_text(yaml.safe_dump({"machine_rank": 0}))

    monkeypatch.setenv("CONFIG_FILE", str(base_config))
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)

    prepare_config()

    output = yaml.safe_load((tmp_path / "acc_cfg.yml").read_text())
    assert output["machine_rank"] == 0
