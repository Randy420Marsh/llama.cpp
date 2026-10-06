set(TARGET llama-tts)
add_executable(${TARGET} tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# fork: the same pipeline kept resident behind an HTTP speech endpoint
set(TARGET llama-tts-server)
add_executable(${TARGET} tts-server.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
