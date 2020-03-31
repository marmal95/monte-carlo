#include <iostream>
#include <random>
#include <iomanip>
#include <chrono>
#include "MpiHelpers.hpp"

struct MonteCarloResult
{
	std::size_t pointsInCircle{};
	std::size_t totalPoints{};

	double calculateRatio() const
	{
		return static_cast<double>(pointsInCircle) / static_cast<double>(totalPoints);
	}
};

template <typename Callable, typename... Args>
auto runWithTimeMeasurementCpu(Callable&& function, Args&&... params)
{
	const auto start = std::chrono::high_resolution_clock::now();
	std::forward<Callable>(function)(std::forward<Args>(params)...);
	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	return duration;
}

double generateRandom()
{
	static std::mt19937 generator{ std::random_device{}() };
	std::uniform_real_distribution<double> distribution(0, std::nextafter(1, std::numeric_limits<double>::max()));
	return distribution(generator);
}

MonteCarloResult monteCarlo(const std::size_t points)
{
	std::size_t pointsInCircle{};

	for (std::size_t index = 0; index < points; ++index)
	{
		const auto randomX = generateRandom();
		const auto randomY = generateRandom();

		const auto distance = std::pow(randomX, 2) + std::pow(randomY, 2);
		if (distance <= 1)
		{
			++pointsInCircle;
		}
	}

	return { pointsInCircle , points };
}

int main()
{
	MPI_Init(nullptr, nullptr);

	MonteCarloResult finalResult{};

	const auto duration = runWithTimeMeasurementCpu([&]() {
		const std::size_t POINTS = 1e9;
		const std::size_t pointsPerProcess = POINTS / MPI::getWorldSize();

		const auto processMonteCarloResult = monteCarlo(pointsPerProcess);
		MPI_Reduce(&processMonteCarloResult.pointsInCircle, &finalResult.pointsInCircle, 1,
			MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI::MASTER_PROCESS, MPI_COMM_WORLD);
		MPI_Reduce(&processMonteCarloResult.totalPoints, &finalResult.totalPoints, 1,
			MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI::MASTER_PROCESS, MPI_COMM_WORLD);
	});
	
	if (MPI::isMasterProcess())
	{
		const auto aproxPi = 4 * finalResult.calculateRatio();
		std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << "PI ~= " << aproxPi
			<< " using " << finalResult.totalPoints << " points." << std::endl;
		std::cout << "Duration [ms]: " << duration << std::endl;
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}