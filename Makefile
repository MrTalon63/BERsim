.PHONY: all sim rebuild clean aff3ct

all: sim

aff3ct:
	@echo "Cleaning, building, and installing AFF3CT..."
	rm -rf aff3ct/build
	mkdir -p aff3ct/build
	cd aff3ct/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DAFF3CT_COMPILE_SHARED_LIB=ON -DAFF3CT_COMPILE_STATIC_LIB=ON -DSPU_STACKTRACE=OFF -DSPU_STACKTRACE_SEGFAULT=OFF
	cd aff3ct/build && make -j$$(nproc)
	@echo "Installing AFF3CT system-wide (requires sudo)..."
	cd aff3ct/build && sudo make -j$$(nproc) install
	@echo "AFF3CT installed! You can now run 'make rebuild'."

sim:
	@mkdir -p build
	@cd build && cmake ..
	@cd build && make -j$$(nproc)

rebuild:
	@echo "Removing old simulator build..."
	rm -rf build
	mkdir -p build
	@echo "Reconfiguring and building simulator..."
	cd build && cmake .. && make -j$$(nproc)

clean:
	rm -rf build