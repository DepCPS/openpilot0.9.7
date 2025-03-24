#!/bin/bash
# Make sure installed wmctrl：sudo apt-get install wmctrl


for i in {1..120}
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
    elif [ $i -ge 21 ] && [ $i -le 30 ]; then
        scenario="3"
        initial_dis="10"
        driver_time="250"
    elif [ $i -ge 31 ] && [ $i -le 40 ]; then
        scenario="4"
        initial_dis="10"
        driver_time="250"
    elif [ $i -ge 41 ] && [ $i -le 50 ]; then
        scenario="5"
        initial_dis="10"
        driver_time="250"
    elif [ $i -ge 51 ] && [ $i -le 60 ]; then
        scenario="6"
        initial_dis="10"
        driver_time="250"
    elif [ $i -ge 61 ] && [ $i -le 70 ]; then
        scenario="1"
        initial_dis="170"
        driver_time="250"
    elif [ $i -ge 71 ] && [ $i -le 80 ]; then
        scenario="2"
        initial_dis="170"
        driver_time="250"
    elif [ $i -ge 81 ] && [ $i -le 90 ]; then
        scenario="3"
        initial_dis="170"
        driver_time="250"
    elif [ $i -ge 91 ] && [ $i -le 100 ]; then
        scenario="4"
        initial_dis="170"
        driver_time="250"
    elif [ $i -ge 101 ] && [ $i -le 110 ]; then
        scenario="5"
        initial_dis="170"
        driver_time="250"
    elif [ $i -ge 111 ] && [ $i -le 120 ]; then
        scenario="6"
        initial_dis="170"
        driver_time="250"
    elif [ $i -ge 121 ] && [ $i -le 130 ]; then
        scenario="1"
        initial_dis="10"
        driver_time="100"
    elif [ $i -ge 131 ] && [ $i -le 140 ]; then
        scenario="2"
        initial_dis="10"
        driver_time="100"
    elif [ $i -ge 141 ] && [ $i -le 150 ]; then
        scenario="3"
        initial_dis="10"
        driver_time="100"
    elif [ $i -ge 151 ] && [ $i -le 160 ]; then
        scenario="4"
        initial_dis="10"
        driver_time="100"
    elif [ $i -ge 161 ] && [ $i -le 170 ]; then
        scenario="5"
        initial_dis="10"
        driver_time="100"
    elif [ $i -ge 171 ] && [ $i -le 180 ]; then
        scenario="6"
        initial_dis="10"
        driver_time="100"
    elif [ $i -ge 181 ] && [ $i -le 190 ]; then
        scenario="1"
        initial_dis="170"
        driver_time="100"
    elif [ $i -ge 191 ] && [ $i -le 200 ]; then
        scenario="2"
        initial_dis="170"
        driver_time="100"
    elif [ $i -ge 201 ] && [ $i -le 210 ]; then
        scenario="3"
        initial_dis="170"
        driver_time="100"
    elif [ $i -ge 211 ] && [ $i -le 220 ]; then
        scenario="4"
        initial_dis="170"
        driver_time="100"
    elif [ $i -ge 221 ] && [ $i -le 230 ]; then
        scenario="5"
        initial_dis="170"
        driver_time="100"
    elif [ $i -ge 231 ] && [ $i -le 240 ]; then
        scenario="6"
        initial_dis="170"
        driver_time="100"
    elif [ $i -ge 241 ] && [ $i -le 250 ]; then
        scenario="1"
        initial_dis="10"
        driver_time="150"
    elif [ $i -ge 251 ] && [ $i -le 260 ]; then
        scenario="2"
        initial_dis="10"
        driver_time="150"
    elif [ $i -ge 261 ] && [ $i -le 270 ]; then
        scenario="3"
        initial_dis="10"
        driver_time="150"
    elif [ $i -ge 271 ] && [ $i -le 280 ]; then
        scenario="4"
        initial_dis="10"
        driver_time="150"
    elif [ $i -ge 281 ] && [ $i -le 290 ]; then
        scenario="5"
        initial_dis="10"
        driver_time="150"
    elif [ $i -ge 291 ] && [ $i -le 300 ]; then
        scenario="6"
        initial_dis="10"
        driver_time="150"
    elif [ $i -ge 301 ] && [ $i -le 310 ]; then
        scenario="1"
        initial_dis="170"
        driver_time="150"
    elif [ $i -ge 311 ] && [ $i -le 320 ]; then
        scenario="2"
        initial_dis="170"
        driver_time="150"
    elif [ $i -ge 321 ] && [ $i -le 330 ]; then
        scenario="3"
        initial_dis="170"
        driver_time="150"
    elif [ $i -ge 331 ] && [ $i -le 340 ]; then
        scenario="4"
        initial_dis="170"
        driver_time="150"
    elif [ $i -ge 341 ] && [ $i -le 350 ]; then
        scenario="5"
        initial_dis="170"
        driver_time="150"
    elif [ $i -ge 351 ] && [ $i -le 360 ]; then
        scenario="6"
        initial_dis="170"
        driver_time="150"
    elif [ $i -ge 361 ] && [ $i -le 370 ]; then
        scenario="1"
        initial_dis="10"
        driver_time="200"
    elif [ $i -ge 371 ] && [ $i -le 380 ]; then
        scenario="2"
        initial_dis="10"
        driver_time="200"
    elif [ $i -ge 381 ] && [ $i -le 390 ]; then
        scenario="3"
        initial_dis="10"
        driver_time="200"
    elif [ $i -ge 391 ] && [ $i -le 400 ]; then
        scenario="4"
        initial_dis="10"
        driver_time="200"
    elif [ $i -ge 401 ] && [ $i -le 410 ]; then
        scenario="5"
        initial_dis="10"
        driver_time="200"
    elif [ $i -ge 411 ] && [ $i -le 420 ]; then
        scenario="6"
        initial_dis="10"
        driver_time="200"
    elif [ $i -ge 421 ] && [ $i -le 430 ]; then
        scenario="1"
        initial_dis="170"
        driver_time="200"
    elif [ $i -ge 431 ] && [ $i -le 440 ]; then
        scenario="2"
        initial_dis="170"
        driver_time="200"
    elif [ $i -ge 441 ] && [ $i -le 450 ]; then
        scenario="3"
        initial_dis="170"
        driver_time="200"
    elif [ $i -ge 451 ] && [ $i -le 460 ]; then
        scenario="4"
        initial_dis="170"
        driver_time="200"
    elif [ $i -ge 461 ] && [ $i -le 470 ]; then
        scenario="5"
        initial_dis="170"
        driver_time="200"
    elif [ $i -ge 471 ] && [ $i -le 480 ]; then
        scenario="6"
        initial_dis="170"
        driver_time="200"
    elif [ $i -ge 481 ] && [ $i -le 490 ]; then
        scenario="1"
        initial_dis="10"
        driver_time="300"
    elif [ $i -ge 491 ] && [ $i -le 500 ]; then
        scenario="2"
        initial_dis="10"
        driver_time="300"
    elif [ $i -ge 501 ] && [ $i -le 510 ]; then
        scenario="3"
        initial_dis="10"
        driver_time="300"
    elif [ $i -ge 511 ] && [ $i -le 520 ]; then
        scenario="4"
        initial_dis="10"
        driver_time="300"
    elif [ $i -ge 521 ] && [ $i -le 530 ]; then
        scenario="5"
        initial_dis="10"
        driver_time="300"
    elif [ $i -ge 531 ] && [ $i -le 540 ]; then
        scenario="6"
        initial_dis="10"
        driver_time="300"
    elif [ $i -ge 541 ] && [ $i -le 550 ]; then
        scenario="1"
        initial_dis="170"
        driver_time="300"
    elif [ $i -ge 551 ] && [ $i -le 560 ]; then
        scenario="2"
        initial_dis="170"
        driver_time="300"
    elif [ $i -ge 561 ] && [ $i -le 570 ]; then
        scenario="3"
        initial_dis="170"
        driver_time="300"
    elif [ $i -ge 571 ] && [ $i -le 580 ]; then
        scenario="4"
        initial_dis="170"
        driver_time="300"
    elif [ $i -ge 581 ] && [ $i -le 590 ]; then
        scenario="5"
        initial_dis="170"
        driver_time="300"
    elif [ $i -ge 591 ] && [ $i -le 600 ]; then
        scenario="6"
        initial_dis="170"
        driver_time="300"
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