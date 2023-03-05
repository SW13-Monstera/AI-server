FROM theeluwin/ubuntu-konlpy:latest

RUN apt-get update

# AWS setting
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY

# copy code
WORKDIR /code
RUN touch ./.git
COPY ./requirements.txt ./requirements.txt


# pytorch install
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install -r requirements.txt

# run FastAPI
COPY ./app ./app
COPY ./script ./script
RUN chmod +x ./script/mecab_install.sh ./script/run.sh

EXPOSE 3000
ENTRYPOINT ["./script/run.sh"]
