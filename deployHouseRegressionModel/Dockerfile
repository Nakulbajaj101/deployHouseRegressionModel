FROM python:3.8-slim

#create a usergroup and user with encrypted password
RUN groupadd fausers && useradd -ms /bin/bash -g fausers -u 58976 ml-user

#chaging the working directory
WORKDIR /opt/ml-api

#adding an environment variable which will come from CI
ARG GEMFURY_PULL_PACKAGE_URL=$GEMFURY_PULL_PACKAGE_URL

#copying the directory and installing requirements
ADD ./housePriceServiceApi ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#giving run privelages to run file
RUN chmod +x ./run.sh

#changing ownership from root to user for the CWD
RUN chown -R ml-user:fausers ./

#chaging the user to run the image
USER ml-user

#exposing the port
EXPOSE 8001

#command to run the image
CMD ["bash", "./run.sh"]
