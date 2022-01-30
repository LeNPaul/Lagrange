FROM ruby:3.1.0-buster
WORKDIR /pvphan.github.io
COPY ./ /pvphan.github.io
RUN bundle install
