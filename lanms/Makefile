# Changed to avoid issues with gcc < 6. Remove '-fno-plt' flag
# CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3-config --cflags)

CXXFLAGS = -I include  -std=c++11 -O3 -I/home/eugene/anaconda3/envs/tfgpu/include/python3.5m -I/home/eugene/anaconda3/envs/tfgpu/include/python3.5m  -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -pipe  -fdebug-prefix-map==/usr/local/src/conda/- -fdebug-prefix-map==/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall -Wstrict-prototypes 

# Run python3-config --ldflags and replace $(shell python3-config --ldflags)
# LDFLAGS = $(shell python3-config --ldflags)
LDFLAGS = -L/home/eugene/anaconda3/envs/tfgpu/lib/python3.5/config-3.5m -L/home/eugene/anaconda3/envs/tfgpu/lib -lpython3.5m -lpthread -ldl  -lutil -lrt -lm  -Xlinker -export-dynamic
DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
