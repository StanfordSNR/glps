enable_testing ()

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --timeout 10 -R '^t_')

add_test(NAME t_alwayspass COMMAND alwayspass)
add_test(NAME t_delay COMMAND delay "${PROJECT_SOURCE_DIR}/etc/t_delay.wisdom")
