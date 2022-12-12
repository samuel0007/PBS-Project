add_executable(SurfaceSampling
	main.cpp
	SamplingBase.h
	SamplingBase.cpp
	SPHSamplingBase.cpp
	SPHSamplingBase.h
	SPHVolumeSampling.cpp
	SPHVolumeSampling.h
	SPHVolumeSampling_Jiang2015.cpp
	SPHVolumeSampling_Jiang2015.h
)
add_dependencies(SurfaceSampling SPlisHSPlasH)
target_link_libraries(SurfaceSampling PRIVATE SPlisHSPlasH)

set_target_properties(SurfaceSampling PROPERTIES FOLDER "Tools")
set_target_properties(SurfaceSampling PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(SurfaceSampling PROPERTIES RELWITHDEBINFO_POSTFIX ${CMAKE_RELWITHDEBINFO_POSTFIX})
set_target_properties(SurfaceSampling PROPERTIES MINSIZEREL_POSTFIX ${CMAKE_MINSIZEREL_POSTFIX})
