LC=ar
LFLAGS=cq

CC=g++
CFLAGS=-c -Wall
INCLUDES=-I/usr/local/include/opencv -I./
LIBRARIES=-L/usr/local/lib/opencv
LDFLAGS=-lm -lcv -lhighgui -lcvaux -lboost_filesystem-mt -lboost_program_options-mt -llog4cxx

LIB_FILES=LCM.cpp SingleLCM.cpp MultipleLCM.cpp
LIB_OFILES=$(LIB_FILES:%.cpp=%.o)
LIB=libhad.a

VIDEOCAPTURE_FILES=VideoCapture.cpp
VIDEOCAPTURE_OFILES=$(VIDEOCAPTURE_FILES:%.cpp=%.o)
VIDEOCAPTURE=videocapture

BACKGROUND_FILES=TestBackground.cpp
BACKGROUND_OFILES=$(BACKGROUND_FILES:%.cpp=%.o)
BACKGROUND=background

COLOR_FILES=TestColor.cpp
COLOR_OFILES=$(COLOR_FILES:%.cpp=%.o)
COLOR=color



all	: $(LIB_FILES) $(VIDEOCAPTURE_FILES) $(BACKGROUND_FILES) $(COLOR_FILES) $(LIB) $(VIDEOCAPTURE) $(BACKGROUND) $(COLOR)

$(LIB):	$(LIB_OFILES)
		rm -f $@
		$(LC) $(LFLAGS) $@ $(LIB_OFILES)

$(VIDEOCAPTURE): $(LIB_OFILES) $(VIDEOCAPTURE_OFILES)
	     	     $(CC) $(INCLUDES) $(LIBRARIES) $(LIB_OFILES) $(VIDEOCAPTURE_OFILES) -o $@ $(LDFLAGS)

$(BACKGROUND): $(LIB_OFILES) $(BACKGROUND_OFILES)
			   $(CC) $(INCLUDES) $(LIBRARIES) $(LIB_OFILES) $(BACKGROUND_OFILES) -o $@ $(LDFLAGS)

$(COLOR): $(LIB_OFILES) $(COLOR_OFILES)
		  $(CC) $(INCLUDES) $(LIBRARIES) $(LIB_OFILES) $(COLOR_OFILES) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBRARIES) $< -o $@ $(LDFLAGS)

clean:
	rm -f ${LIB_OFILES} ${VIDEOCAPTURE_OFILES} ${BACKGROUND_OFILES} ${COLOR_OFILES} *~
			   
