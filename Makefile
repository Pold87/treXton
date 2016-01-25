CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
ALLFLAGS = $(CFLAGS) $(LIBS) -std=c++11

all: treXton

treXton: treXton.o
	g++ -g treXton.o -o treXton $(ALLFLAGS)

treXton.o: treXton.cpp
	g++ -g -c treXton.cpp $(ALLFLAGS)

clean:
	rm *.o treXton
