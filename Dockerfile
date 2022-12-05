FROM jekyll/jekyll:4.2.0
# until release of 4.3 we should ping 4.2.0
Label MAINTAINER Amir Pourmand
#install imagemagick tool for convert command
RUN apk add --no-cache --virtual .build-deps \
        libxml2-dev \
        shadow \
        autoconf \
        g++ \
        make
#    && apk add --no-cache imagemagick-dev imagemagick
WORKDIR /srv/jekyll
ADD Gemfile /srv/jekyll/
ADD lagrange.gemspec /srv/jekyll
RUN bundle install
RUN bundle add webrick