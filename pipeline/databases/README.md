# Setting up MySQL 8.0 and MongoDB 4.4 on Ubuntu 20.04 LTS

This guide provides a comprehensive step-by-step approach to installing and running MySQL 8.0 and MongoDB 4.4 on Ubuntu 20.04 LTS. Additionally, it covers importing SQL dumps and essential configuration.

## Requirements
- **Operating System**: Ubuntu 20.04 LTS
- **MySQL Version**: 8.0
- **MongoDB Version**: 4.4

---

## 1. Installing MySQL 8.0

### Step 1: Update the System
```bash
sudo apt-get update
```

### Step 2: Install MySQL Server
```bash
sudo apt-get install mysql-server
```

### Step 3: Secure MySQL Installation
Run the secure installation script:
```bash
sudo mysql_secure_installation
```

### Step 4: Configure MySQL Root User
Access the MySQL shell:
```bash
sudo mysql
```
Set the root password and enable native authentication:
```sql
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'your_password';
FLUSH PRIVILEGES;
EXIT;
```

---

## 2. Running MySQL and Importing a SQL Dump

### Step 1: Start the MySQL Service
```bash
sudo service mysql start
```

### Step 2: Create a Database
```bash
echo "CREATE DATABASE hbtn_0d_tvshows;" | mysql -uroot -p
```

### Step 3: Import the SQL Dump
```bash
curl "https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows.sql" -s | mysql -uroot -p hbtn_0d_tvshows
```

### Step 4: Verify the Import
```bash
echo "SELECT * FROM tv_genres;" | mysql -uroot -p hbtn_0d_tvshows
```

---

## 3. Writing SQL Files with Comments
Example of a SQL file with comments:

```sql
-- Retrieve the 3 most recent students in Batch ID=3
-- Batch 3 is known for excellence!
SELECT id, name FROM students WHERE batch_id = 3 ORDER BY created_at DESC LIMIT 3;
```

Save the above in a file named `my_script.sql`.

---

## 4. Installing MongoDB 4.4

### Step 1: Import the MongoDB Public Key
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
```

### Step 2: Create MongoDB Repository
```bash
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list
```

### Step 3: Update the Package Database
```bash
sudo apt-get update
```

### Step 4: Install MongoDB
```bash
sudo apt-get install -y mongodb-org
```

---

## 5. Running MongoDB

### Step 1: Start the MongoDB Service
```bash
sudo service mongod start
```

### Step 2: Verify MongoDB Installation
```bash
sudo service mongod status
```

### Step 3: Start the MongoDB Shell
```bash
mongosh
```

To list databases:
```bash
echo "show dbs" | mongosh
```

---

## 6. Common Issues and Fixes

### Issue: Data Directory Not Found
If you encounter the error: `Data directory /data/db not found., terminating`

Fix it by creating the directory:
```bash
sudo mkdir -p /data/db
sudo chown -R `id -u` /data/db
```

---

## 7. Installing and Verifying PyMongo

### Step 1: Install PyMongo
```bash
pip3 install pymongo
```

### Step 2: Verify Installation
```bash
python3
>>> import pymongo
>>> pymongo.__version__
'4.6.2'
```

---

## Notes
- All SQL keywords should be in uppercase.
- The first line of all MongoDB scripts should be:
  ```
  // my comment
  ```
- Python files should start with:
  ```bash
  #!/usr/bin/env python3
  

