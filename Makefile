WORKDIR_PATH=/pvphan.github.io
REPO_PATH:=$(dir $(abspath $(firstword $(MAKEFILE_LIST))))
IMAGE_TAG=pvphan/blog:latest
RUN_FLAGS = --rm \
	--network=host \
	${IMAGE_TAG}


serve: image
	docker run ${RUN_FLAGS} \
		jekyll serve --livereload

shell: image
	docker run -it ${RUN_FLAGS} \
		bash

image:
	docker build --tag ${IMAGE_TAG} .

