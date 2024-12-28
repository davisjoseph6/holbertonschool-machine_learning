# Setting Up MongoDB 8.0 Community Edition on Ubuntu

This guide provides step-by-step instructions for installing MongoDB 8.0 Community Edition on supported Ubuntu systems using the official MongoDB packages.

## Supported Ubuntu Versions

MongoDB 8.0 Community Edition supports the following 64-bit Ubuntu LTS releases:
- Ubuntu 24.04 LTS ("Noble")
- Ubuntu 22.04 LTS ("Jammy")
- Ubuntu 20.04 LTS ("Focal")

To check your Ubuntu version, run:
```bash
cat /etc/lsb-release
```

---

## Prerequisites

1. Ensure your system is updated:
   ```bash
   sudo apt-get update
   ```
2. Install `gnupg` and `curl` (if not already installed):
   ```bash
   sudo apt-get install -y gnupg curl
   ```

---

## Step 1: Import the MongoDB Public GPG Key

Run the following command to add the MongoDB GPG key:
```bash
curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor
```

---

## Step 2: Create the MongoDB Repository List File

### Ubuntu 24.04 ("Noble")
```bash
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu noble/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list
```

### Ubuntu 22.04 ("Jammy")
Replace `noble` with `jammy`:
```bash
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list
```

### Ubuntu 20.04 ("Focal")
Replace `noble` with `focal`:
```bash
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list
```

---

## Step 3: Reload the Package Database

Update the local package database to include the MongoDB repository:
```bash
sudo apt-get update
```

---

## Step 4: Install MongoDB Community Edition

To install MongoDB, run:
```bash
sudo apt-get install -y mongodb-org
```

---

## Step 5: Start and Enable MongoDB

### Start the MongoDB Service
```bash
sudo systemctl start mongod
```

### Enable MongoDB to Start on Boot
```bash
sudo systemctl enable mongod
```

---

## Step 6: Verify MongoDB Installation

Check the status of the MongoDB service:
```bash
sudo systemctl status mongod
```

Start the MongoDB shell to verify installation:
```bash
mongosh
```
You should see a welcome message and the MongoDB shell prompt.

---

## Additional Commands

### Stop MongoDB
```bash
sudo systemctl stop mongod
```

### Restart MongoDB
```bash
sudo systemctl restart mongod
```

### Check Logs
Logs are available at:
```bash
/var/log/mongodb/mongod.log
```

---

## Directories and Configuration

- **Data Directory**: `/var/lib/mongodb`
- **Log Directory**: `/var/log/mongodb`
- **Configuration File**: `/etc/mongod.conf`

Changes to the configuration file take effect after restarting MongoDB:
```bash
sudo systemctl restart mongod
```

---

## Start Using MongoDB

Run the following command to start a MongoDB shell session:
```bash
mongosh
```
### Example:
```bash
> show dbs
```
This command displays the databases available in MongoDB.

---

## Troubleshooting

If you encounter issues during installation or setup, refer to:
- [MongoDB Troubleshooting Guide](https://www.mongodb.com/docs/manual/administration/production-notes/)

---

Congratulations! You have successfully installed MongoDB 8.0 Community Edition on Ubuntu.

