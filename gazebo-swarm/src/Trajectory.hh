#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace swarm
{
struct Sample
{
  double time;
  // x, y, z, unwrapped yaw (radians).
  std::array<double, 4> pose;
};

class Trajectory
{
 public:
  explicit Trajectory(const std::string &path)
  {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open trajectory: " + path);
    std::string line;
    if (!std::getline(input, line) || line != "time,x,y,z,yaw")
      throw std::runtime_error("Invalid CSV header: " + path);
    while (std::getline(input, line))
    {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream row(line);
      Sample sample{};
      if (!(row >> sample.time >> sample.pose[0] >> sample.pose[1]
                >> sample.pose[2] >> sample.pose[3]))
        throw std::runtime_error("Invalid CSV row: " + path);
      std::string extra;
      if (row >> extra) throw std::runtime_error("Extra CSV column: " + path);
      if (!std::isfinite(sample.time) || sample.time < 0 ||
          (!samples.empty() && sample.time <= samples.back().time))
        throw std::runtime_error("Invalid sample time: " + path);
      for (double value : sample.pose)
        if (!std::isfinite(value))
          throw std::runtime_error("Non-finite pose: " + path);
      samples.push_back(sample);
    }
    if (samples.size() < 2 || samples.front().time != 0)
      throw std::runtime_error("Trajectory must start at zero and have >=2 rows");
  }

  std::array<double, 4> At(double time, bool interpolate = true) const
  {
    if (time <= samples.front().time) return samples.front().pose;
    if (time >= samples.back().time) return samples.back().pose;
    auto next = std::upper_bound(samples.begin(), samples.end(), time,
        [](double t, const Sample &sample) { return t < sample.time; });
    const auto &previous = *(next - 1);
    if (!interpolate) return previous.pose;
    double alpha = (time - previous.time) / (next->time - previous.time);
    std::array<double, 4> result{};
    for (std::size_t i = 0; i < result.size(); ++i)
      result[i] = previous.pose[i] + alpha * (next->pose[i] - previous.pose[i]);
    return result;
  }

 private:
  std::vector<Sample> samples;
};
}  // namespace swarm
