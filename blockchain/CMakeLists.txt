cmake_minimum_required(VERSION 3.1...3.20)

project(Blockchain
        DESCRIPTION "blockchain for storing news"
        LANGUAGES CXX
)

include(FetchContent)

add_executable(${PROJECT_NAME}
    main.cpp
    Peer.cpp
    block.cpp
    blockchain.cpp
    hash.cpp
)

FetchContent_Declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG master
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)

FetchContent_MakeAvailable(asio)

FetchContent_Declare(
    cargs
    GIT_REPOSITORY https://github.com/likle/cargs.git
    GIT_TAG stable
    SOURCE_DIR CARGS
)

FetchContent_MakeAvailable(cargs)
target_link_libraries(${PROJECT_NAME} PUBLIC cargs)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${asio_SOURCE_DIR}/asio/include)

find_package(Threads)
find_package(OpenSSL REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenSSL::Crypto Threads::Threads Qt6::Widgets)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PUBLIC ASIO_STANDALONE PURIFY)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)
#target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR "clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Ofast -Wall -Wextra -Wpedantic)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
endif()
