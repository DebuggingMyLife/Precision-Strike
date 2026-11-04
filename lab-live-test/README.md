# Lab live test

This folder is a standalone copy of the live-test tools. The `models`
subfolder contains the checkpoints needed by the default commands.

`run_sim_live_test.py` loads the current `drone_soccer_ppo_v4.1.zip` policy and
the trained `tello_defender_goal` YOLO weights, then runs them together in the
PyBullet GUI environment. It sends no commands to a physical drone.

From this extracted folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements-live-test.txt
python .\run_sim_live_test.py
```

`run_tello_yolo_preview.py` is the physical-camera check. It connects to the
Tello, displays the trained detector's boxes, and never takes off or sends RC
commands:

```powershell
python .\run_tello_yolo_preview.py
```

The PPO policy is not sent directly to a real Tello here. Its four actions are
simulated motor controls, while Tello's API accepts velocity commands, and the
policy's distance feature needs a real-camera calibration. Those conversions
must be validated separately before flight control is enabled.


