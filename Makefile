CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
ALLFLAGS = $(CFLAGS) $(LIBS) -std=c++11

all: ground_truth

ground_truth: ground_truth.cpp
	g++ -o ground_truth ground_truth.cpp relocalize.h relocalize.cpp $(CFLAGS) $(LIBS) -lboost_system -std=c++11


# treXton: treXton.o
# 	g++ -g treXton.o -o treXton $(ALLFLAGS)

# treXton.o: treXton.cpp
# 	g++ -g -c treXton.cpp $(ALLFLAGS)

clean:
	rm *.o treXton ground_truth
