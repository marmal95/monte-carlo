#include "Algorithm.hpp"
#include <iostream>

int main()
{
	const auto[result, duration] = runWithTimeMeasurementCpu(monteCarlo);

	std::cout.precision(std::numeric_limits<double>::max_digits10);
	std::cout << "PI ~= " << 4 * result.calculateRatio() << std::endl;
	std::cout << "Estimated with total points: " << result.totalPoints << std::endl;
	std::cout << "Duration [ms]: " << duration << std::endl;

	return EXIT_SUCCESS;
}
