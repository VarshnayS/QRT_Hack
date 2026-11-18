#include "main.cpp"

#include <cstdlib>
#include <stdexcept>

namespace test {
using Grid = std::vector<std::string>;
using Point = std::pair<int, int>;
constexpr std::array<Point, 4> Step{{{-1, 0}, {0, 1}, {1, 0}, {0, -1}}};

void require(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(reason);
}

bool turn(Point a, Point b, Point c) {
    return a.first != c.first && a.second != c.second && a != b && b != c;
}

char eligible(const std::vector<Point>& loop, int i) {
    const int n = int(loop.size());
    auto corner = [&](int k) {
        return turn(loop[(k + n - 1) % n], loop[k], loop[(k + 1) % n]);
    };
    const bool before = corner((i + n - 1) % n), after = corner((i + 1) % n);
    if (corner(i)) return !before && !after ? 'B' : '.';
    return before || after ? 'W' : '.';
}

int check(const Grid& clues, const std::string& output) {
    const int rows = int(clues.size()), cols = int(clues[0].size());
    require(int(output.size()) == rows * cols, "missing or malformed solver output");
    const std::string alphabet = ".-|LJF7";
    const std::array<std::string, 7> directions{"", "13", "02", "01", "03", "12", "23"};
    auto links = [&](Point pt) -> const std::string& {
        const auto state = alphabet.find(output[pt.first * cols + pt.second]);
        require(state != std::string::npos, "unknown output glyph");
        return directions[state];
    };
    const int first = int(output.find_first_not_of('.'));
    require(first >= 0, "empty loop");
    const Point start{first / cols, first % cols};
    Point cur = start;
    int incoming = -1;
    std::vector<Point> loop;
    Grid seen(rows, std::string(cols, '.'));
    do {
        require(seen[cur.first][cur.second] == '.', "repeated cell");
        seen[cur.first][cur.second] = '#';
        loop.push_back(cur);
        const auto& dirs = links(cur);
        require(dirs.size() == 2, "walk entered an empty cell");
        const int d = dirs[0] - '0' == incoming ? dirs[1] - '0' : dirs[0] - '0';
        cur.first += Step[d].first;
        cur.second += Step[d].second;
        require(cur.first >= 0 && cur.first < rows && cur.second >= 0 && cur.second < cols,
                "connection leaves the board");
        incoming = (d + 2) % 4;
        require(links(cur).find(char('0' + incoming)) != std::string::npos, "nonreciprocal edge");
    } while (cur != start);
    for (int i = 0; i < int(loop.size()); ++i) {
        const auto [r, c] = loop[i];
        require(clues[r][c] == '.' || clues[r][c] == eligible(loop, i), "pearl rule violated");
    }
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if (output[r * cols + c] != '.' || clues[r][c] != '.')
                require(seen[r][c] == '#', "disconnected loop or missed pearl");
    return int(loop.size());
}

// Enumerate every simple cycle shorter than a known witness, rooted at one pearl.
int oracle(const Grid& clues, int upper) {
    const int rows = int(clues.size()), cols = int(clues[0].size());
    std::vector<Point> pearls;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if (clues[r][c] != '.') pearls.emplace_back(r, c);
    const Point start = pearls.front();
    std::vector<Point> loop{start};
    Grid seen(rows, std::string(cols, '.'));
    seen[start.first][start.second] = '#';
    auto walk = [&](auto&& self, Point current, int remaining) -> void {
        const int length = int(loop.size());
        if (length + remaining >= upper) return;
        for (const auto& [dr, dc] : Step) {
            const Point next{current.first + dr, current.second + dc};
            const auto [r, c] = next;
            if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
            if (next == start) {
                if (remaining || length < 4 || loop[1] > loop.back()) continue;
                bool valid = true;
                for (int i = 0; i < length; ++i) {
                    const auto [a, b] = loop[i];
                    if (clues[a][b] != '.' && clues[a][b] != eligible(loop, i)) {
                        valid = false;
                        break;
                    }
                }
                if (valid) upper = length;
                continue;
            }
            if (seen[r][c] == '#') continue;
            const int distance = std::abs(r - start.first) + std::abs(c - start.second);
            if (length + distance >= upper) continue;
            if (length >= 2) {
                const char clue = clues[current.first][current.second];
                if (clue != '.' && (clue == 'B') != turn(loop[length - 2], current, next))
                    continue;
            }
            seen[r][c] = '#';
            loop.push_back(next);
            self(self, next, remaining - (clues[r][c] != '.'));
            loop.pop_back();
            seen[r][c] = '.';
        }
    };
    walk(walk, start, int(pearls.size()) - 1);
    return upper;
}

struct Generated { Grid clues; int witness; };
Generated generate(int rows, int cols, int density, int growth, std::mt19937& rng) {
    const int bottom = growth < 0 ? 2 : rows - 1;
    const int right = growth < 0 ? 2 : cols - 1;
    std::vector<Point> loop;
    for (int c = 0; c < right; ++c) loop.emplace_back(0, c);
    for (int r = 0; r < bottom; ++r) loop.emplace_back(r, right);
    for (int c = right; c > 0; --c) loop.emplace_back(bottom, c);
    for (int r = bottom; r > 0; --r) loop.emplace_back(r, 0);
    Grid used(rows, std::string(cols, '.'));
    for (auto [r, c] : loop) used[r][c] = '#';
    for (int step = 0; step < std::abs(growth); ++step) {
        const int k = int(rng() % loop.size());
        const auto [ar, ac] = loop[k];
        const auto [br, bc] = loop[(k + 1) % loop.size()];
        const int sign = rng() & 1 ? 1 : -1;
        const int dr = (bc - ac) * sign, dc = (ar - br) * sign;
        const Point a{ar + dr, ac + dc}, b{br + dr, bc + dc};
        auto free = [&](Point point) {
            const auto [r, c] = point;
            return r >= 0 && r < rows && c >= 0 && c < cols && used[r][c] == '.';
        };
        if (!free(a) || !free(b)) continue;
        used[a.first][a.second] = used[b.first][b.second] = '#';
        loop.insert(loop.begin() + k + 1, {a, b});
    }
    Grid clues(rows, std::string(cols, '.'));
    std::vector<int> candidates;
    for (int i = 0; i < int(loop.size()); ++i)
        if (eligible(loop, i) != '.') candidates.push_back(i);
    require(candidates.size() >= 2, "generator needs at least two pearl candidates");
    std::shuffle(candidates.begin(), candidates.end(), rng);
    const int count = std::max(2, int(candidates.size()) * density / 100);
    for (int i = 0; i < count; ++i) {
        const int k = candidates[i];
        const auto [r, c] = loop[k];
        clues[r][c] = eligible(loop, k);
    }
    return {clues, int(loop.size())};
}

void run(const Grid& clues, int expected, int witness, int index) {
    const Puzzle p(clues);
    const auto start = Clock::now();
    const Grid solution = solve(clues);
    try {
        require(solution.size() == clues.size(), "incorrect output row count");
        std::string flattened;
        for (const auto& row : solution) {
            require(row.size() == clues.front().size(), "incorrect output column count");
            flattened += row;
        }
        const int length = check(clues, flattened);
        if (expected) require(length == expected, "disagrees with exact optimum");
        std::cout << index << ' ' << p.rows << 'x' << p.cols << " pearls=" << p.pearls.size()
                  << " length=" << length << " witness=" << witness
                  << " total=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    } catch (...) {
        std::cerr << "Case " << index << '\n' << p.rows << '\n';
        for (const auto& row : clues) std::cerr << row << '\n';
        if (expected) std::cerr << "Expected length " << expected << '\n';
        throw;
    }
}

void testBooleanEngine() {
    std::mt19937 random(10839);
    constexpr int variables = 8;
    for (int trial = 0; trial < 100; ++trial) {
        BooleanSearch sat(variables);
        std::vector<std::vector<int>> clauses;
        for (int batch = 0; batch < 3; ++batch) {
            sat.reset();
            for (int k = 0; k < 14; ++k) {
                std::vector<int> clause;
                int used = 0;
                while (clause.size() < 3) {
                    const int v = int(random() % variables);
                    if (used & (1 << v)) continue;
                    used |= 1 << v;
                    clause.push_back(2 * v + int(random() & 1));
                }
                clauses.push_back(clause);
                sat.constrain(std::move(clause));
            }
            const auto satisfies = [&](int value) {
                for (const auto& clause : clauses) {
                    bool satisfied = false;
                    for (int lit : clause)
                        satisfied |= ((value >> (lit / 2)) ^ lit) & 1;
                    if (!satisfied) return false;
                }
                return true;
            };
            bool expected = false;
            for (int value = 0; value < (1 << variables); ++value) expected |= satisfies(value);
            sat.initializeOrder();
            const bool found = sat.solve(Clock::now() + std::chrono::seconds(1),
                                         [] { return std::vector<int>{}; }, 23);
            require(expected == found, "Boolean search disagrees with exhaustive truth table");
            if (found) {
                int value = 0;
                for (int i = 0; i < variables; ++i)
                    if (sat.on(i)) value |= 1 << i;
                require(satisfies(value), "Boolean search returned an invalid assignment");
            }
        }
    }
}
}  // namespace test

int main(int argc, char** argv) {
    using namespace test;
    testBooleanEngine();
    const Grid subtour{".W....", "......", "......", "....W.", "......", "......"};
    const std::string disjoint = "F-7..." "|.|..." "L-J..." "...F-7" "...|.|" "...L-J";
    require(Puzzle(subtour).validate(disjoint) == 0, "submission checker accepted two cycles");
    bool rejected = false;
    try { check(subtour, disjoint); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "test checker accepted two cycles");
    const Grid sample{"......", ".W....", "....B.", "......", ".B....", "....W."};
    require(check(sample, "......" "F-7..." "L7L-7." ".|..|." ".L-7L7" "...L-J") == 18,
            "invalid sample witness");
    require(oracle(sample, 18) == 18, "sample has a shorter cycle");
    int failures = 0;
    const int first = argc > 2 ? std::stoi(argv[2]) : 0;
    const int last = argc > 3 ? std::stoi(argv[3]) : 75;
    auto verify = [&](const Grid& clues, int expected, int witness, int index) {
        if (index < first || index > last) return;
        try {
            run(clues, expected, witness, index);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            ++failures;
        }
    };
    verify(sample, 18, 18, 0);
    std::mt19937 rng(argc > 1 ? unsigned(std::stoul(argv[1])) : 1924u);
    for (int i = 1; i <= 24; ++i) {
        const int growth = -int(1 + rng() % 22), density = 10 + int(rng() % 91);
        const auto g = generate(6, 6, density, growth, rng);
        verify(g.clues, oracle(g.clues, g.witness), g.witness, i);
    }
    for (int i = 25; i <= 51; ++i) {
        const std::array<Point, 5> sizes{{{6, 50}, {50, 6}, {15, 20}, {30, 35}, {50, 50}}};
        const auto [rows, cols] = sizes[(i - 25) % sizes.size()];
        const std::array<int, 3> densities{3, 36, 98};
        const auto g = generate(rows, cols, densities[(i - 25) % 3], rows * cols * 3, rng);
        verify(g.clues, 0, g.witness, i);
    }
    for (int i = 52; i <= 63; ++i) {
        const int growth = -int(1 + rng() % 22), density = 10 + int(rng() % 91);
        const auto g = generate(i % 2 ? 7 : 6, 12, density, growth, rng);
        verify(g.clues, oracle(g.clues, g.witness), g.witness, i);
    }
    for (int i = 64; i <= 75; ++i) {
        const int density = 15 + int(rng() % 86);
        const auto g = generate(6, 6, density, 24 + i, rng);
        verify(g.clues, oracle(g.clues, g.witness), g.witness, i);
    }
    return failures ? 1 : 0;
}
