#!/usr/bin/env bash
# services/setup.sh — one-time setup on the services VM
# Prerequisites:
#   - block volume mounted at /mnt/db-data
#   - EC2_ACCESS_KEY and EC2_SECRET_KEY exported
set -e

if [ -z "$EC2_ACCESS_KEY" ] || [ -z "$EC2_SECRET_KEY" ]; then
    echo "ERROR: EC2_ACCESS_KEY and EC2_SECRET_KEY must be set"
    exit 1
fi

echo "=== Installing Docker ==="
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker

echo "=== Writing .env ==="
cat > .env << EOF
EC2_ACCESS_KEY=${EC2_ACCESS_KEY}
EC2_SECRET_KEY=${EC2_SECRET_KEY}
OBJSTORE_CONTAINER=ObjStore_proj21
GENERATOR_ARRIVAL_RATE=500.0
GENERATOR_INITIAL_USERS=5
EOF

echo "=== Setup complete. Run: docker compose up -d --build ==="
