PYTHON ?= python
CPPFLAGS := -I..
CPPFLAGS += -I$(shell $(PYTHON) -c "import distutils.sysconfig as c; print(c.get_python_inc())")
CFLAGS   := -std=c11 -fPIC -g
LDFLAGS  := -shared -L.. -g
#LIBS     := -lndtypes

all: examples.so

examples.so: examplesmodule.o gufuncs.o infrastructure.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

check: examples.so
	$(PYTHON) -m pytest -sv test_examples.py

clean:
	rm -rf *.o *.so
