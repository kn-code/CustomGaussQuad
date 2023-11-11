project(
    'CustomGaussQuad',
    'cpp',
    version : '1.0',
    default_options : ['warning_level=3',
                       'cpp_std=c++20',
                       'buildtype=release',
                       'optimization=2'
                      ])

eigen_dep = dependency('eigen3', fallback : ['eigen', 'eigen_dep'])
mpfr_dep = dependency('mpfr', required : false)

#
# Tests
#

boostut = subproject('boost.ut')
boostut_dep = boostut.get_variable('boostut_dep')

quadrature_test = executable(
    'quadrature_test',
    ['tests/quadrature_test.cpp'],
    dependencies: [eigen_dep, boostut_dep]
    )
test('quadrature_test', quadrature_test)


if mpfr_dep.found()
    mpfr_quadrature_test = executable(
        'mpfr_quadrature_test',
        ['tests/mpfr_quadrature_test.cpp'],
        dependencies : [eigen_dep, mpfr_dep, boostut_dep]
        )
    test('mpfr_quadrature_test', mpfr_quadrature_test)
endif

#
# Install
#

CustomGaussQuad_dep = declare_dependency(
                include_directories : include_directories('.'),
                dependencies: [eigen_dep, mpfr_dep])


install_headers('custom_gauss_quad.h')

pkgc = import('pkgconfig')
pkgc.generate(
    name: 'CustomGaussQuad',
    description: 'Computation of Gaussian quadrature rules with custom weight functions in C++',
    version: meson.project_version(),
    subdirs: ['.', 'eigen3']
)