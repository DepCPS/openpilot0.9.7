#!/bin/bash
# Make sure installed wmctrl：sudo apt-get install wmctrl


for i in {1..20}
do
    echo "=== #$i ==="

    if [ $i -ge 1 ] && [ $i -le 10 ]; then
        scenario="1"
        initial_dis="10"
        driver_time="250"
    elif [ $i -ge 11 ] && [ $i -le 20 ]; then
        scenario="2"
        initial_dis="10"
        driver_time="250"
    else
        scenario="0"
        initial_dis="0"
        driver_time="0"
    fi

    echo "Now：--scenario ${scenario} --initial_dis ${initial_dis} --driver_time ${driver_time}"

    gnome-terminal --title="openpilot-$i" -- bash -c "cd tools/sim && poetry run ./launch_openpilot.sh; exec bash"

    gnome-terminal --title="bridge-$i" -- bash -c "cd tools/sim && poetry run ./run_bridge.py --scenario ${scenario} --initial_dis ${initial_dis} --driver_time ${driver_time}; exec bash"

    sleep 120

    wmctrl -c "openpilot-$i"
    wmctrl -c "bridge-$i"

    sleep 5
done