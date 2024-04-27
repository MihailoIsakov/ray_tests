#! /bin/bash

HEAD=172.20.52.43
PORT=6379
ssh root@172.20.52.43 "source ./ray/.env/bin/activate && ray stop && ray start --head --port=${PORT} --dashboard-host=0.0.0.0"
ssh root@172.20.52.42 "source ./ray/.env/bin/activate && ray stop && ray start --address=${HEAD}:${PORT}" & \
ssh root@172.20.52.49 "source ./ray/.env/bin/activate && ray stop && ray start --address=${HEAD}:${PORT}" & \
ssh root@172.20.52.50 "source ./ray/.env/bin/activate && ray stop && ray start --address=${HEAD}:${PORT}" &
