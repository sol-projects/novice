cmake_minimum_required(VERSION 3.1...3.20)

project(Blockchain
        DESCRIPTION "blockchain for storing news"
        LANGUAGES CXX
)

include(FetchContent)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

add_executable(${PROJECT_NAME}
    main.cpp
    Peer.cpp
    block.cpp
    blockchain.cpp
    hash.cpp
    service.cpp
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

FetchContent_Declare(httplib SYSTEM
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(httplib)

find_package(Threads)
find_package(OpenSSL REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenSSL::Crypto Threads::Threads Qt6::Widgets httplib)

find_package(MPI REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PUBLIC ASIO_STANDALONE PURIFY)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR "clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -g -Wall -Wextra -Wpedantic)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
endif()
