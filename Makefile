help:
	@cat Makefile
SRC?=$(shell pwd)
torch:
	docker build -t torch .
notebook: torch
	docker run -it -v ${SRC}/src:/data --net=host torch
