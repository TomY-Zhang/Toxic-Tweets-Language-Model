# Installation

## Docker
1. Go to the [Docker website](https://www.docker.com/products/docker-desktop/) and install the latest version of Docker Desktop compatible with your operating system. For reference, I am using Docker Desktop v4.17.0 for MacOS.

2. To ensure that Docker is properly installed, open your terminal and run `docker version`.<br>
This will display the properties of Docker version installed.

3. Navigate to your project directory and run `touch Dockerfile` to create a Dockerfile.

4. Run `touch app.js` to create a JavaScript file. Open it with your preferred text editor or IDE and enter `console.log('Hello Docker')`.

5. Open the Dockerfile.<br>
a. Enter `FROM node:alpine` on the first line. This command specifies `node:alpine` as the base image used to construct a Docker image.<br>
b. Enter `COPY . /app` on the second line. When a Docker container is created, all files in the project directory (aside from the Dockerfile) will be copied into the `/app` directory of the container.<br>
c. Enter `WORKDIR /app` on the third line to specify the default woring directory of the Docker container.<br>
d. Enter `CMD node app.js` on the fourth line. When the Docker container is created, it wil automatically execute this command.<br>

6. To build the Docker image, run `docker build -t hello-docker .` in the terminal.

7. Execute `docker run hello-docker` to create a container from the `hello-docker` image.