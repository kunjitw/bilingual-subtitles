"""重複啟動與 port（app/instance.py、server.main）的測試：真的啟動伺服器子程序。

全部用暫存的 data 資料夾（VS_DATA_DIR）、空的 models 資料夾、不存在的 llama-server 位置，
port 只用 8821 到 8839 裡沒人用的，測完用 PID 結束這個測試自己開的伺服器，暫存資料夾整個刪掉。
不碰正式伺服器（8765）和正式資料庫。會跑佇列的那個伺服器，佇列裡沒有能執行的任務，不會用到顯示卡。

python -s tests/test_instance.py
"""
import http.server
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

from app import instance  # noqa: E402

BASE = Path(tempfile.gettempdir()) / "instance-test"
PY = sys.executable
PORTS = range(8821, 8840)
_used: set[int] = set()


def _port_free(port: int) -> bool:
    try:
        instance.bind_socket("127.0.0.1", port).close()
        return True
    except OSError:
        return False


def take_port(pair: bool = False) -> int:
    """8821 到 8839 裡沒人用的 port（pair=True 時連下一個也要沒人用）。"""
    for p in PORTS:
        if p in _used or not _port_free(p) or (pair and (p + 1 in _used or p + 1 not in PORTS or not _port_free(p + 1))):
            continue
        _used.update({p, p + 1} if pair else {p})
        return p
    raise RuntimeError("8821 到 8839 沒有空的 port，沒辦法測試")


class Lab:
    """一個暫存資料夾，裡面可以開好幾個伺服器（共用或不共用 data 資料夾）。"""

    def __init__(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="vs-inst-", dir=BASE))
        self.procs: list[subprocess.Popen] = []
        self.logs: list[Path] = []

    def start(self, data: Path, args=(), workers=False, port=None) -> tuple[subprocess.Popen, Path]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("VS_")}
        env.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", VS_DATA_DIR=str(data), VS_MODELS_DIR=str(self.tmp / "models"),
                   VS_LLAMA_SERVER=str(self.tmp / "no-llama" / "llama-server.exe"))
        if not workers:
            env["VS_NO_WORKERS"] = "1"
        if port is not None:
            env["VS_PORT"] = str(port)
        log = self.tmp / f"server-{len(self.logs)}.log"
        self.logs.append(log)
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.Popen([PY, "-s", "-m", "app.server", "--no-browser", *args], cwd=str(ROOT), env=env,
                                    stdout=f, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        self.procs.append(proc)
        return proc, log

    def close(self):
        for proc in self.procs:          # 只結束這個測試自己開的伺服器
            if proc.poll() is None:
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        child.kill()
                    proc.kill()
                except psutil.Error:
                    pass
                proc.wait(timeout=30)
        time.sleep(0.5)
        shutil.rmtree(self.tmp, ignore_errors=True)


def run(fn):
    lab = Lab()
    try:
        fn(lab)
    finally:
        lab.close()


def wait_ready(port: int, proc: subprocess.Popen, log: Path, timeout: float = 90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"伺服器提早結束（{proc.returncode}）：{log.read_text('utf-8', errors='replace')}")
        if instance.is_ours("127.0.0.1", port, timeout=1.0):
            return
        time.sleep(0.3)
    raise AssertionError(f"伺服器沒有在 {timeout} 秒內啟動：{log.read_text('utf-8', errors='replace')}")


def wait_exit(proc: subprocess.Popen, log: Path, timeout: float = 90) -> tuple[int, str]:
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"應該要結束卻還在跑：{log.read_text('utf-8', errors='replace')}")
    return code, log.read_text("utf-8", errors="replace")


def listening(pid: int) -> set[tuple[str, int]]:
    return {(c.laddr.ip, c.laddr.port) for c in psutil.Process(pid).net_connections(kind="tcp")
            if c.status == psutil.CONN_LISTEN}


class _NotOurs(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_port_taken_by_other_program_moves_to_next_and_binds_localhost_only():
    def body(lab: Lab):
        port = take_port(pair=True)
        other = http.server.HTTPServer(("127.0.0.1", port), _NotOurs)     # 別的程式占住 port
        th = threading.Thread(target=other.serve_forever, daemon=True)
        th.start()
        try:
            proc, log = lab.start(lab.tmp / "data-a", port=port)
            wait_ready(port + 1, proc, log)
            text = log.read_text("utf-8", errors="replace")
            assert f"http://127.0.0.1:{port + 1}/" in text and f"改用 {port + 1}" in text, text
            # 預設（沒設 VS_HOST、區網開關關著）只聽 127.0.0.1
            assert listening(proc.pid) == {("127.0.0.1", port + 1)}, listening(proc.pid)

            # --port 明確指定、被占用：直接結束並說明，不自己換 port
            proc2, log2 = lab.start(lab.tmp / "data-b", args=["--port", str(port)])
            code, text2 = wait_exit(proc2, log2)
            assert code == 1 and "已經被其他程式占用" in text2, (code, text2)
        finally:
            other.shutdown()
            other.server_close()
    run(body)


def test_second_start_on_same_port_opens_existing_and_exits():
    def body(lab: Lab):
        port = take_port()
        first, log = lab.start(lab.tmp / "data", port=port)
        wait_ready(port, first, log)
        second, log2 = lab.start(lab.tmp / "data", port=port)
        code, text = wait_exit(second, log2)
        assert code == 0 and f"Bilingual Subtitles 已經在執行：http://127.0.0.1:{port}/" in text, (code, text)
        assert first.poll() is None and instance.is_ours("127.0.0.1", port)
        # 指定同一個 port：報錯
        third, log3 = lab.start(lab.tmp / "data", args=["--port", str(port)])
        code, text = wait_exit(third, log3)
        assert code == 1 and "已經有 Bilingual Subtitles 在執行" in text, (code, text)
    run(body)


def _jobs_db(data: Path) -> sqlite3.Connection:
    con = sqlite3.connect(data / "library.db", timeout=10, isolation_level=None)
    con.row_factory = sqlite3.Row
    return con


def test_only_one_server_runs_the_queue():
    def body(lab: Lab):
        data = lab.tmp / "data"
        port, port2, port3 = take_port(), take_port(), take_port()
        first, log = lab.start(data, workers=True, port=port)
        wait_ready(port, first, log)
        info = json.loads((data / "server.json").read_text("utf-8"))
        assert info["port"] == port and info["pid"] == first.pid, info

        # 第一個伺服器啟動後才有一個執行中的任務（例如正在轉字幕）
        con = _jobs_db(data)
        con.execute("INSERT INTO jobs (id, media_id, type, params, status, stage, position, created_at, started_at) "
                    "VALUES ('aaaaaaaaaaaa', NULL, 'transcribe', '{}', 'running', '辨識中', 1, ?, ?)",
                    (time.time(), time.time()))

        # 第二次啟動（連點兩次 start.bat）：拿不到鎖，打開第一個的網頁後結束，佇列不動
        second, log2 = lab.start(data, workers=True, args=["--port", str(port2)])
        code, text = wait_exit(second, log2)
        assert code == 0 and f"Bilingual Subtitles 已經在執行：http://127.0.0.1:{port}/" in text, (code, text)
        assert not instance.is_ours("127.0.0.1", port2, timeout=0.5)
        row = con.execute("SELECT status, stage FROM jobs WHERE id = 'aaaaaaaaaaaa'").fetchone()
        assert (row["status"], row["stage"]) == ("running", "辨識中"), dict(row)

        # 只提供網頁的測試伺服器（VS_NO_WORKERS=1）：不鎖、不改佇列、不改 server.json
        third, log3 = lab.start(data, args=["--port", str(port3)])
        wait_ready(port3, third, log3)
        row = con.execute("SELECT status FROM jobs WHERE id = 'aaaaaaaaaaaa'").fetchone()
        assert row["status"] == "running"
        assert json.loads((data / "server.json").read_text("utf-8"))["pid"] == first.pid

        # 不經過 main() 直接啟動 app（例如 TestClient、uvicorn 指令）：lifespan 拿不到鎖就不跑佇列
        script = ("import json; from app import jobs, server; called = []; "
                  "jobs.start_workers = lambda: called.append(1); "
                  "from fastapi.testclient import TestClient\n"
                  "with TestClient(server.app) as c: c.get('/api/meta')\n"
                  "print(json.dumps({'called': bool(called)}))")
        env = {k: v for k, v in os.environ.items() if not k.startswith("VS_")}
        env.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", VS_DATA_DIR=str(data),
                   VS_MODELS_DIR=str(lab.tmp / "models"))
        out = subprocess.run([PY, "-s", "-c", script], cwd=str(ROOT), env=env, capture_output=True, text=True,
                             encoding="utf-8", timeout=120)
        assert out.returncode == 0 and json.loads(out.stdout.strip().splitlines()[-1]) == {"called": False}, out.stderr
        row = con.execute("SELECT status FROM jobs WHERE id = 'aaaaaaaaaaaa'").fetchone()
        assert row["status"] == "running"

        # 第一個結束之後，鎖就放開了：下一個會跑佇列的可以拿到
        first.kill()
        first.wait(timeout=30)
        con.close()
        out = subprocess.run([PY, "-s", "-c", script], cwd=str(ROOT), env=env, capture_output=True, text=True,
                             encoding="utf-8", timeout=120)
        assert out.returncode == 0 and json.loads(out.stdout.strip().splitlines()[-1]) == {"called": True}, out.stderr
    run(body)


def test_reserved_ports_are_skipped_unless_explicit():
    """WinError 10013（Hyper-V、WSL 保留的 port 範圍）：預設換下一個，--port 明確指定時說明後結束。不真的綁 port。"""
    real_bind, real_ours = instance.bind_socket, instance.is_ours
    tried = []

    def fake_bind(host, port):
        tried.append(port)
        if port in (8765, 8766):
            raise OSError(None, "存取被拒", None, 10013)
        if port == 8767:
            raise OSError(None, "位址已被使用", None, 10048)
        return "sock"

    instance.bind_socket, instance.is_ours = fake_bind, lambda host, port, timeout=2.0: False
    try:
        assert instance.open_port("127.0.0.1", 8765, explicit=False) == ("sock", 8768) and tried == [8765, 8766, 8767, 8768]
        for port, words in ((8765, "被 Windows 保留"), (8767, "已經被其他程式占用")):
            try:
                instance.open_port("127.0.0.1", port, explicit=True)
            except instance.StartupStop as stop:
                assert stop.code == 1 and words in str(stop) and stop.url is None, str(stop)
            else:
                raise AssertionError("明確指定的 port 不能用時應該結束")
        instance.bind_socket = lambda host, port: (_ for _ in ()).throw(OSError(None, "x", None, 10048))
        try:
            instance.open_port("127.0.0.1", 8765, explicit=False)
        except instance.StartupStop as stop:
            assert "8765 到 8790" in str(stop), str(stop)
        else:
            raise AssertionError("全部被占用時應該結束")
    finally:
        instance.bind_socket, instance.is_ours = real_bind, real_ours


def test_bind_socket_detects_wildcard_listener():
    """別的程式綁在 0.0.0.0 同一個 port 時，Windows 仍然允許綁 127.0.0.1，要算成被占用。"""
    port = take_port()
    other = socket.socket()
    other.bind(("0.0.0.0", port))
    other.listen()
    try:
        try:
            instance.bind_socket("127.0.0.1", port).close()
        except OSError as e:
            assert instance._in_use(e) or instance._forbidden(e), e
        else:
            raise AssertionError("0.0.0.0 被占用時不應該綁得到 127.0.0.1")
    finally:
        other.close()
    assert instance.browser_url("0.0.0.0", 8765) == "http://127.0.0.1:8765/"
    assert instance.browser_url("192.168.1.5", 8766) == "http://192.168.1.5:8766/"
    assert instance.browser_url("::1", 8767) == "http://[::1]:8767/"         # ::1 只收 IPv6，127.0.0.1 連不到
    assert instance.browser_url("::", 8768) == "http://127.0.0.1:8768/"
    assert instance.local_only("127.0.0.1") and instance.local_only("::1") and instance.local_only("localhost")
    assert not instance.local_only("0.0.0.0") and not instance.local_only("::") and not instance.local_only("192.168.1.5")


def test_ipv6_hosts_are_reachable_at_printed_url():
    """VS_HOST=:: 綁成同時收 IPv4、IPv6（Windows 預設只收 IPv6），印出來的 127.0.0.1 網址連得到；::1 用 [::1]。"""
    if not socket.has_ipv6:
        print("  （這台電腦沒有 IPv6，跳過）")
        return
    for host in ("::", "::1"):
        port = take_port()
        sock = instance.bind_socket(host, port)
        try:
            sock.listen()
            url = instance.browser_url(host, port)
            target = ("127.0.0.1", port) if "127.0.0.1" in url else ("::1", port)
            family = socket.AF_INET if target[0] == "127.0.0.1" else socket.AF_INET6
            with socket.socket(family, socket.SOCK_STREAM) as c:
                c.settimeout(3)
                c.connect(target)
        finally:
            sock.close()


if __name__ == "__main__":
    started = time.time()
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print("ok", name)
    finally:
        if BASE.exists() and not any(BASE.iterdir()):
            BASE.rmdir()
    print(f"all passed in {time.time() - started:.1f}s")
