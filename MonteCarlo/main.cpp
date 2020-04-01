#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <limits>

template <typename Callable, typename... Args>
auto runWithTimeMeasurementCpu(Callable&& function, Args&&... params)
{
	const auto start = std::chrono::high_resolution_clock::now();
	const auto result = std::forward<Callable>(function)(std::forward<Args>(params)...);
	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	return std::make_pair(result, duration);
}

struct MonteCarloResult
{
	std::size_t pointsInCircle{};
	std::size_t totalPoints{};

	double calculateRatio() const
	{
		return static_cast<double>(pointsInCircle) / static_cast<double>(totalPoints);
	}
};

double generateRandom()
{
	static std::mt19937 generator{ std::random_device{}() };
	std::uniform_real_distribution<double> distribution(0, std::nextafter(1, std::numeric_limits<double>::max()));
	return distribution(generator);
}

MonteCarloResult monteCarlo()
{
	std::random_device rd{};
	std::mt19937 randomEngine(rd());
	std::uniform_real_distribution<double> dist(0, std::nextafter(1, std::numeric_limits<double>::max()));

	const std::size_t POINTS = 1e10;
	std::size_t pointsInCircle{};

	static std::size_t counter{};
	long long index{};

	#pragma omp parallel for private(index) schedule(dynamic) reduction(+ : pointsInCircle)
	for (index = 0; index < POINTS; ++index)
	{
		const auto randomX = generateRandom();
		const auto randomY = generateRandom();

		const auto distance = std::pow(randomX, 2) + std::pow(randomY, 2);
		if (distance <= 1)
		{
			++pointsInCircle;
		}
	}

	return { pointsInCircle , POINTS };
}

int main()
{
	const auto[monteCarloResult, duration] = runWithTimeMeasurementCpu(monteCarlo);
	const auto aproxPi = 4 * monteCarloResult.calculateRatio();

	std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << "PI ~= " << aproxPi
		<< " using " << monteCarloResult.totalPoints << " points." << std::endl;
	std::cout << "Duration [ms]: " << duration << std::endl;

	return EXIT_SUCCESS;
}