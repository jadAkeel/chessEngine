from app.infra.config import AppConfig, SystemConfig
from app.infra.runtime import resolve_thread_plan


def test_render_free_api_uses_one_inference_thread_unless_overridden(monkeypatch):
    cfg = AppConfig(system=SystemConfig(cpu_threads=4))
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_CPU_COUNT", "0.1")
    monkeypatch.delenv("CPU_THREADS", raising=False)
    assert resolve_thread_plan(cfg, device="cpu", role="api").intra_threads == 1

    monkeypatch.setenv("CPU_THREADS", "4")
    assert resolve_thread_plan(cfg, device="cpu", role="api").intra_threads == 4

    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("CPU_THREADS", raising=False)
    assert resolve_thread_plan(cfg, device="cpu", role="api").intra_threads == 4
