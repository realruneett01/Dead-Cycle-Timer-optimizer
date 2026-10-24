# Platform verification

What has and has not been verified, with the exact commands to close the gaps. Each block lists what
"pass" looks like. Run from the repository root; `python` means Python 3.11 or 3.12.

## Verified so far (Linux x86-64, 4 vCPU)

| Area | Evidence |
|---|---|
| Unit/integration tests | 39 pass (`pytest -q`) |
| Docker | Both images build (`dcto:full` 944 MB, `dcto:edge` 512 MB). Container runs as non-root, streams telemetry to the `/data` volume, dashboard answers HTTP 200, healthcheck goes `healthy`, memory limit (`--memory=1g`) and CPU quota (`--cpus=1`) both select the `edge` profile, and the container exits 1 when its simulator dies so `restart: unless-stopped` recovers it |
| systemd unit | `systemd-analyze verify deploy/dcto.service` is clean. **Not run under a live systemd.** |
| Windows | Wheels exist for every dependency on win_amd64, Python 3.11 and 3.12. **Not executed.** |
| ARM | Wheels exist for every dependency on manylinux aarch64, Python 3.11 and 3.12. **Not executed.** 32-bit ARM is not checked and likely lacks wheels (see below). |
| Pendrive | Offline install from `vendor/wheels`, demo, full commission, quick verify, run; on Linux only |

## 1. Windows 10/11 (PowerShell)

```powershell
py -3.12 --version                      # or: py -3.11 --version
.\install.ps1                            # builds .venv, writes config\dcto.json
.\.venv\Scripts\python.exe -m deploy doctor
.\.venv\Scripts\python.exe -m pytest -q
.\dcto.bat --demo                        # whole pipeline against the built-in simulator
```
Pass: `doctor` exits 0 with no `BLOCKER`; pytest reports all passed; the demo ends with `COMMISSIONING PASSED`.

Windows-specific risks to watch: Ctrl+C shutdown leaving a stray `python.exe` (check Task Manager after
`.\.venv\Scripts\python.exe -m deploy run --no-dashboard`), the memory probe in `doctor` (`memory_mb`
should match Task Manager), and Windows Defender Firewall prompts for ports 4840 and 8501.

## 2. ARM

On a 64-bit OS (Raspberry Pi OS 64-bit, Ubuntu arm64, Apple Silicon Linux VM):
```bash
uname -m                                  # aarch64
sh dcto.sh --demo
python3 -m deploy doctor | grep -E '"(arch|profile|memory_mb|cpu_count)"'
python3 tests/measure_latency.py          # the only real edge-hardware latency figure you will have
```
Pass: demo ends `COMMISSIONING PASSED`; on a Pi with 2 GB or less `profile` is `edge`.

Docker on any host, to build the arm64 image (needs buildx and QEMU, or an arm64 machine):
```bash
docker buildx build --platform linux/arm64 -t dcto:arm64 --load .
docker run --rm --platform linux/arm64 dcto:arm64 python -m deploy doctor
```
32-bit ARM (`armv7l`): the profile logic selects `edge`, but PyPI has no armv7l wheels for scipy/pandas,
so the install will compile from source or use piwheels. Not recommended; use a 64-bit OS.

## 3. Docker

```bash
docker build -t dcto:full .
docker build --build-arg DASHBOARD=0 -t dcto:edge .
docker run --rm dcto:full python -m deploy doctor
docker run --rm --memory=1g --cpus=1 dcto:full python -m deploy doctor | grep profile   # expect "edge"

docker run -d --name dcto -p 8501:8501 -v dcto-data:/data dcto:full
sleep 60
docker ps                                 # STATUS should read (healthy)
docker exec dcto python -m deploy health && echo healthy
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501/_stcore/health   # 200
docker logs dcto | grep -E "Streaming|ANOMALY|Traceback"
docker rm -f dcto

docker compose config -q && DCTO_ENDPOINT=opc.tcp://host:4840 docker compose up -d
```
Pass: healthy, HTTP 200, no `Traceback`. With a real `DCTO_ENDPOINT` the container will stay unhealthy
and exit with code 2 until the plant has been commissioned (that is the gate working); commission first
with `docker run -it --rm -v dcto-data:/data -e DCTO_ENDPOINT=... dcto:full python -m deploy commission`.

## 4. systemd (Linux host with systemd as PID 1)

```bash
sudo ./install.sh --service
systemctl status dcto --no-pager          # active (running)
journalctl -u dcto -n 40 --no-pager       # look for "Streaming ... -> detector"
sudo -u root .venv/bin/python -m deploy health && echo healthy

# Restart on failure: kill the simulator child, systemd should bring the service back within ~5 s
sudo pkill -f opcua.opcua_server; sleep 8; systemctl is-active dcto

sudo reboot                               # then: systemctl is-active dcto   (enabled at boot)
sudo systemctl disable --now dcto && sudo rm /etc/systemd/system/dcto.service
```
Note: a service pointed at a real plant (`DCTO_ENDPOINT` in the unit's environment) will refuse to start
until commissioned: run `.venv/bin/python -m deploy commission --endpoint ...` once first.

## 5. Pendrive end to end

```bash
python -m deploy bundle                   # on a machine of the SAME OS/CPU/Python as the target
# copy the whole folder to the stick, plug into the target, then:
sh dcto.sh --demo                         # or dcto.bat --demo on Windows
sh dcto.sh --endpoint opc.tcp://<plc-gateway>:4840      # first time on the real plant
sh dcto.sh                                # later plug-ins: quick verify, then run
```
Test a write-protected stick too: commissioning should fail with `data-dir-writable`, not crash.
