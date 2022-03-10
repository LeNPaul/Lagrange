FROM ruby:3.1.0-buster

COPY ./Gemfile /tmp/
COPY ./lagrange.gemspec /tmp/
WORKDIR /tmp
RUN bundle install

WORKDIR /pvphan.github.io
