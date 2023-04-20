# Setup Server to run Docker

steps to run on server to get app running 

[reference](http://kitware.github.io/paraviewweb/docs/docker.html)

##Required Packages

### Installation

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

sudo apt-get install apache2-dev apache2 libapr1-dev apache2-utils
```

reboot suggested
```
sudo reboot
```

### Docker runtime install

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
  "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) \
  stable"

sudo apt-get update
sudo apt-get install docker-ce

sudo systemctl restart docker
```


## Apache Configuration

```
sudo a2enmod vhost_alias
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod proxy_wstunnel
sudo a2enmod rewrite
sudo a2enmod headers

sudo service apache2 restart
```


make a proxy server `/etc/apache2/sites-available/uncertainsci-web-demo.conf`


```
<VirtualHost *:80>
  ServerName   ${SERVER_NAME}
  Redirect permanent / https://${SERVER_NAME}/
  DocumentRoot /home/ubuntu/UncertainSCI-web-demo_old/bundles/docker/server/www/
  ErrorLog ${APACHE_LOG_DIR}/error.log
  CustomLog ${APACHE_LOG_DIR}/access.log combined
</VirtualHost>
```

also add servername to `/etc/apache2/apache2.conf`

```
ServerName myserver.mydomain.com
```


add new site, disable default
```
sudo a2dissite 000-default.conf
sudo a2ensite uncertainsci-web-demo
sudo systemctl reload apache2
```

## Build app

### install packages

```
sudo apt install python3-pip python3 python3-venv
```


### build app

get source code and build
```
git clone https://github.com/SCIInstitute/UncertainSCI-web-demo.git
cd UncertainSCI-web-demo

python3 -m venv venv
source venv/bin/activate

pip install .
```

### build and deploy docker

```
cd bundles/docker
sudo ./scripts/build_server.sh

sudo ./scripts/run_server.sh
```










