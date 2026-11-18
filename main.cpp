#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Mask = uint8_t;
constexpr int States = 7, All = 127, Empty = 1;
constexpr std::array<int, States> Edges{0, 10, 5, 3, 9, 6, 12};
constexpr char Glyph[] = ".-|LJF7";
constexpr int Straight = 6, Turn = 120;
constexpr int Inf = 100000;
constexpr int opposite(int direction) { return (direction + 2) & 3; }
int bitCount(int mask) {
    int count = 0;
    for (; mask; mask &= mask - 1) ++count;
    return count;
}

struct Puzzle {
    int rows, cols, n;
    std::string clues;
    std::vector<std::array<int, 4>> adj;
    std::vector<int> pearls;
    std::array<int, 4> uses{};
    std::array<int, 128> possible{}, forced{}, count{}, singleton{};
    std::array<int, 16> allowed{};
    std::vector<Mask> initial;

    explicit Puzzle(const std::vector<std::string>& grid)
        : rows(int(grid.size())), cols(int(grid.front().size())), n(rows * cols),
          adj(n), initial(n) {
        for (int s = 0; s < States; ++s)
            for (int d = 0; d < 4; ++d)
                if (Edges[s] & (1 << d)) uses[d] |= 1 << s;
        for (int m = 1; m <= All; ++m) {
            count[m] = bitCount(m);
            for (int s = 0; s < States; ++s)
                if (m & (1 << s)) singleton[m] = s;
            for (int d = 0; d < 4; ++d) {
                if (m & uses[d]) possible[m] |= 1 << d;
                if (!(m & (All ^ uses[d]))) forced[m] |= 1 << d;
            }
        }
        for (int edges = 0; edges < 16; ++edges)
            for (int s = 0; s < States; ++s)
                if ((Edges[s] & edges) == Edges[s]) allowed[edges] |= 1 << s;
        for (const auto& row : grid) clues += row;
        for (int i = 0; i < n; ++i) {
            const int r = i / cols, c = i % cols;
            adj[i] = {r ? i - cols : -1, c + 1 < cols ? i + 1 : -1,
                      r + 1 < rows ? i + cols : -1, c ? i - 1 : -1};
            int mask = All;
            if (clues[i] != '.') {
                pearls.push_back(i);
                mask = clues[i] == 'B' ? Turn : Straight;
            }
            for (int d = 0; d < 4; ++d)
                if (adj[i][d] < 0) mask &= All ^ uses[d];
            initial[i] = Mask(mask);
        }
    }

    int validate(const std::string& grid) const {
        if (int(grid.size()) != n) return 0;
        // Independently decode the output alphabet into north, east, south, west.
        std::vector<int> links(n);
        auto decode = [](char ch) {
            switch (ch) {
                case '.': return 0;
                case '-': return 2 | 8;
                case '|': return 1 | 4;
                case 'L': return 1 | 2;
                case 'J': return 1 | 8;
                case 'F': return 4 | 2;
                case '7': return 4 | 8;
                default: return -1;
            }
        };
        int first = -1, length = 0;
        for (int i = 0; i < n; ++i) {
            links[i] = decode(grid[i]);
            if (links[i] < 0 || (!links[i] && clues[i] != '.')) return 0;
            if (links[i]) {
                if (bitCount(links[i]) != 2) return 0;
                first = i;
                ++length;
            }
        }
        if (first < 0) return 0;
        auto straight = [](int mask) { return mask == 5 || mask == 10; };
        for (int i = 0; i < n; ++i) {
            int straightNeighbors = 0;
            for (int d = 0; d < 4; ++d) {
                if (!(links[i] & (1 << d))) continue;
                const int j = adj[i][d];
                if (j < 0 || !(links[j] & (1 << opposite(d)))) return 0;
                straightNeighbors += straight(links[j]);
            }
            if (clues[i] == 'W' && (!straight(links[i]) || straightNeighbors == 2)) return 0;
            if (clues[i] == 'B' && (straight(links[i]) || straightNeighbors != 2)) return 0;
        }
        std::vector<char> visited(n);
        std::vector<int> walk{first};
        visited[first] = 1;
        for (size_t k = 0; k < walk.size(); ++k) {
            const int i = walk[k];
            for (int d = 0; d < 4; ++d) {
                if (!(links[i] & (1 << d))) continue;
                const int j = adj[i][d];
                if (!visited[j]) {
                    visited[j] = 1;
                    walk.push_back(j);
                }
            }
        }
        return int(walk.size()) == length ? length : 0;
    }
};

// Small incremental CDCL engine used when domain search encounters a difficult
// combination of pearl orientations. Literals are 2 * variable + negation.
class BooleanSearch {
    struct Clause { std::vector<int> lits; };
    std::vector<Clause> clauses;
    std::vector<std::vector<int>> watches;
    std::vector<int8_t> value, phase;
    std::vector<int> level, reason, trail, limits;
    std::vector<char> seen;
    std::vector<double> activity;
    std::priority_queue<std::pair<double, int>> order;
    size_t head = 0;
    size_t storedLiterals = 0;
    double increment = 1;
    bool feasible = true;

    int eval(int literal) const {
        const int v = value[literal / 2];
        return v < 0 ? -1 : v ^ (literal & 1);
    }

    void assign(int literal, int because) {
        const int v = literal / 2;
        value[v] = phase[v] = int8_t(1 ^ (literal & 1));
        reason[v] = because;
        level[v] = int(limits.size());
        trail.push_back(literal);
    }

    int add(std::vector<int> literals) {
        const int id = int(clauses.size());
        storedLiterals += literals.size();
        clauses.push_back({std::move(literals)});
        const auto& lits = clauses.back().lits;
        if (lits.size() >= 2) {
            watches[lits[0]].push_back(id);
            watches[lits[1]].push_back(id);
        }
        return id;
    }

    int propagate() {
        while (head < trail.size()) {
            const int dead = trail[head++] ^ 1;
            auto& list = watches[dead];
            size_t read = 0, write = 0;
            while (read < list.size()) {
                const int id = list[read++];
                auto& lits = clauses[id].lits;
                if (lits[0] == dead) std::swap(lits[0], lits[1]);
                const int first = lits[0];
                if (eval(first) != 1) {
                    size_t k = 2;
                    while (k < lits.size() && eval(lits[k]) == 0) ++k;
                    if (k < lits.size()) {
                        lits[1] = lits[k];
                        lits[k] = dead;
                        watches[lits[1]].push_back(id);
                        continue;
                    }
                    if (eval(first) == 0) {
                        list[write++] = id;
                        while (read < list.size()) list[write++] = list[read++];
                        list.resize(write);
                        return id;
                    }
                    assign(first, id);
                }
                list[write++] = id;
            }
            list.resize(write);
        }
        return -1;
    }

    void rebuildOrder() {
        order = {};
        for (int i = 0; i < int(value.size()); ++i)
            if (value[i] < 0) order.emplace(activity[i], i);
    }

    void backtrack(int target) {
        if (int(limits.size()) <= target) return;
        const size_t keep = limits[target];
        while (trail.size() > keep) {
            const int v = trail.back() / 2;
            value[v] = -1;
            order.emplace(activity[v], v);
            trail.pop_back();
        }
        head = std::min(head, keep);
        limits.resize(target);
    }

    int learn(int conflict, std::vector<int>& learned) {
        learned.assign(1, -1);
        int open = 0, resolved = -1;
        size_t pos = trail.size();
        do {
            for (int lit : clauses[conflict].lits) {
                const int v = lit / 2;
                if (v == resolved || seen[v] || !level[v]) continue;
                seen[v] = 1;
                activity[v] += increment;
                if (value[v] < 0) order.emplace(activity[v], v);
                if (level[v] == int(limits.size())) ++open;
                else learned.push_back(lit);
            }
            do { resolved = trail[--pos] / 2; } while (!seen[resolved]);
            seen[resolved] = 0;
            --open;
            conflict = reason[resolved];
        } while (open);
        learned[0] = trail[pos] ^ 1;
        increment *= 1.045;
        if (increment > 1e85) {
            increment *= 1e-85;
            for (double& a : activity) a *= 1e-85;
            rebuildOrder();
        }
        size_t highest = 1;
        for (size_t i = 1; i < learned.size(); ++i) {
            seen[learned[i] / 2] = 0;
            if (level[learned[i] / 2] > level[learned[highest] / 2]) highest = i;
        }
        if (learned.size() == 1) return 0;
        std::swap(learned[1], learned[highest]);
        return level[learned[1] / 2];
    }

public:
    explicit BooleanSearch(int vars)
        : watches(2 * vars), value(vars, -1), phase(vars), level(vars), reason(vars),
          seen(vars), activity(vars) {}

    void prefer(int variable, double priority) { activity[variable] = priority; }
    void initializeOrder() { rebuildOrder(); }
    bool on(int variable) const { return value[variable] == 1; }
    bool off(int variable) const { return value[variable] == 0; }
    void reset() { backtrack(0); }

    void constrain(std::vector<int> literals) {
        if (!feasible) return;
        size_t count = 0;
        for (int literal : literals) {
            const int result = eval(literal);
            if (result == 1) return;
            if (result < 0) literals[count++] = literal;
        }
        literals.resize(count);
        if (!count) feasible = false;
        else if (count == 1) assign(literals[0], -1);
        else add(std::move(literals));
    }

    template<class Connectivity>
    bool solve(Clock::time_point stop, Connectivity connectivity, unsigned interval) {
        if (!feasible) return false;
        int conflicts = 0, nextRestart = 100, restart = 0;
        uint64_t steps = 0;
        std::vector<int> learned;
        while ((++steps & 63) != 0 ||
               (Clock::now() < stop && clauses.size() < 350000 && storedLiterals < 8000000)) {
            int conflict = propagate();
            if (conflict < 0 && steps % interval == 0 && !limits.empty()) {
                auto cut = connectivity();
                if (!cut.empty()) {
                    std::partial_sort(cut.begin(), cut.begin() + std::min<size_t>(2, cut.size()),
                                      cut.end(), [&](int a, int b) { return level[a / 2] > level[b / 2]; });
                    backtrack(level[cut.front() / 2]);
                    conflict = add(std::move(cut));
                }
            }
            if (conflict >= 0) {
                if (limits.empty()) { feasible = false; return false; }
                const int target = learn(conflict, learned);
                backtrack(target);
                const int unit = learned.front();
                const int id = learned.size() == 1 ? -1 : add(learned);
                assign(unit, id);
                if (++conflicts == nextRestart) {
                    backtrack(0);
                    nextRestart += 85 + 15 * (++restart % 16);
                }
                continue;
            }
            if (order.size() > value.size() * 12) rebuildOrder();
            while (!order.empty() &&
                   (value[order.top().second] >= 0 || activity[order.top().second] != order.top().first))
                order.pop();
            if (order.empty()) return true;
            const int v = order.top().second;
            order.pop();
            limits.push_back(int(trail.size()));
            assign(2 * v + (1 ^ phase[v]), -1);
        }
        return false;
    }
};

class Solver {
    const Puzzle& p;
    std::vector<Mask> domain;
    struct Change { int cell; Mask old; };
    std::vector<Change> trail;
    std::vector<int> pending;
    std::vector<char> queued;
    std::array<int, 2> required{};
    std::vector<int> parent, size, degree, active, endpoints, distance, bfs;
    std::vector<int> discovery, low, predecessor, cursor, kept, vertexStack, edgeStack;
    std::mt19937 rng{0x4d617379};
    Clock::time_point start = Clock::now(), deadline;
    int mode = 0;
    int geometricBound = 0;
    double elapsed() const { return std::chrono::duration<double>(Clock::now() - start).count(); }

    int color(int i) const { return (i / p.cols + i % p.cols) & 1; }
    int manhattan(int a, int b) const {
        return std::abs(a / p.cols - b / p.cols) + std::abs(a % p.cols - b % p.cols);
    }
    void enqueue(int i) {
        if (!queued[i]) {
            queued[i] = 1;
            pending.push_back(i);
        }
    }

    bool narrow(int i, int support) {
        const int old = domain[i], now = old & support;
        if (!now) return false;
        if (old == now) return true;
        trail.push_back({i, Mask(old)});
        domain[i] = Mask(now);
        if ((old & Empty) && !(now & Empty)) ++required[color(i)];
        enqueue(i);
        for (int j : p.adj[i])
            if (j >= 0 && p.clues[j] != '.') enqueue(j);
        return true;
    }

    void restore(size_t mark) {
        for (int i : pending) queued[i] = 0;
        pending.clear();
        while (trail.size() > mark) {
            const auto change = trail.back();
            trail.pop_back();
            if ((change.old & Empty) && !(domain[change.cell] & Empty))
                --required[color(change.cell)];
            domain[change.cell] = change.old;
        }
    }

    bool pearl(int i) {
        const bool black = p.clues[i] == 'B';
        int centerSupport = 0;
        std::array<int, 4> support{};
        for (int s = 1; s < States; ++s) {
            if (!(domain[i] & (1 << s))) continue;
            std::array<int, 4> match{};
            int turns = 0;
            bool valid = true;
            for (int d = 0; d < 4; ++d) {
                const int j = p.adj[i][d];
                if (j < 0) continue;
                const int reverse = p.uses[opposite(d)];
                if (Edges[s] & (1 << d)) {
                    match[d] = domain[j] & reverse & (black ? Straight : All);
                    if (match[d] & Turn) turns |= 1 << d;
                } else {
                    match[d] = domain[j] & (All ^ reverse);
                }
                if (!match[d]) { valid = false; break; }
            }
            if (!valid || (!black && !turns)) continue;
            centerSupport |= 1 << s;
            for (int d = 0; d < 4; ++d)
                support[d] |= !black && turns == (1 << d) ? match[d] & Turn : match[d];
        }
        if (!narrow(i, centerSupport)) return false;
        for (int d = 0; d < 4; ++d) {
            const int j = p.adj[i][d];
            if (j >= 0 && !narrow(j, support[d])) return false;
        }
        return true;
    }

    bool propagate() {
        while (!pending.empty()) {
            const int i = pending.back();
            pending.pop_back();
            queued[i] = 0;
#ifdef MASYU_STATS
            ++propagations;
#endif
            const int possible = p.possible[domain[i]], forced = p.forced[domain[i]];
            for (int d = 0; d < 4; ++d) {
                const int j = p.adj[i][d], bit = 1 << d;
                if (j < 0) continue;
                const int reverse = p.uses[opposite(d)];
                if (!(possible & bit) && !narrow(j, All ^ reverse)) return false;
                if ((forced & bit) && !narrow(j, reverse)) return false;
            }
            if (p.clues[i] != '.' && !pearl(i)) return false;
        }
        return true;
    }

    // A simple cycle lies wholly in one vertex-biconnected block.
    // The block containing all required cells is unique because there are at least two pearls.
    bool restrictBlock() {
        std::fill(discovery.begin(), discovery.end(), 0);
        std::fill(kept.begin(), kept.end(), 0);
        vertexStack.clear();
        edgeStack.clear();
        const int root = p.pearls.front();
        int timer = 1;
        discovery[root] = low[root] = timer;
        predecessor[root] = -1;
        cursor[root] = 0;
        vertexStack.push_back(root);
        bool found = false;
        while (!vertexStack.empty() && !found) {
            const int i = vertexStack.back();
            if (cursor[i] < 4) {
                const int d = cursor[i]++;
                if (!(p.possible[domain[i]] & (1 << d))) continue;
                const int j = p.adj[i][d];
                if (!discovery[j]) {
                    discovery[j] = low[j] = ++timer;
                    predecessor[j] = i;
                    cursor[j] = 0;
                    vertexStack.push_back(j);
                    edgeStack.push_back(i * 4 + d);
                } else if (j != predecessor[i] && discovery[j] < discovery[i]) {
                    low[i] = std::min(low[i], discovery[j]);
                    edgeStack.push_back(i * 4 + d);
                }
                continue;
            }
            vertexStack.pop_back();
            const int par = predecessor[i];
            if (par < 0) continue;
            low[par] = std::min(low[par], low[i]);
            if (low[i] < discovery[par]) continue;
            int requiredHere = 0;
            const size_t end = edgeStack.size();
            size_t begin = end;
            bool terminal;
            do {
                const int edge = edgeStack[--begin];
                const int a = edge / 4, d = edge % 4, b = p.adj[a][d];
                kept[a] |= 1 << d;
                kept[b] |= 1 << opposite(d);
                terminal = a == par && b == i;
            } while (!terminal);
            for (size_t k = begin; k < end; ++k) {
                const int edge = edgeStack[k], a = edge / 4, b = p.adj[a][edge % 4];
                if (kept[a] > 0) {
                    requiredHere += !(domain[a] & Empty);
                    kept[a] = -kept[a];
                }
                if (kept[b] > 0) {
                    requiredHere += !(domain[b] & Empty);
                    kept[b] = -kept[b];
                }
            }
            found = requiredHere == required[0] + required[1] && end - begin >= 4;
            if (!found) {
                for (size_t k = begin; k < end; ++k) {
                    const int edge = edgeStack[k];
                    kept[edge / 4] = kept[p.adj[edge / 4][edge % 4]] = 0;
                }
            }
            edgeStack.resize(begin);
        }
        if (!found) return false;
        for (int i = 0; i < p.n; ++i)
            if (!narrow(i, p.allowed[-kept[i]])) return false;
        return propagate();
    }

    int root(int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }

    // Returns a member of a closed component, or -1 for a forest of paths.
    int components() {
        active.clear();
        endpoints.clear();
        for (int i = 0; i < p.n; ++i) {
            if (domain[i] & Empty) continue;
            parent[i] = i;
            size[i] = 1;
            degree[i] = bitCount(p.forced[domain[i]]);
            active.push_back(i);
            if (degree[i] < 2) endpoints.push_back(i);
        }
        for (int i : active) {
            const int edges = p.forced[domain[i]];
            for (int d = 1; d <= 2; ++d) {
                if (!(edges & (1 << d))) continue;
                int a = root(i), b = root(p.adj[i][d]);
                if (a == b) return a;
                if (size[a] < size[b]) std::swap(a, b);
                parent[b] = a;
                size[a] += size[b];
            }
        }
        return -1;
    }

    bool record(std::string candidate) {
        const int length = p.validate(candidate);
        if (!length) return false;
        if (length >= bestLength) return true;
        if (best.empty()) firstTime = elapsed();
        bestLength = length;
        best = std::move(candidate);
        ++improvements;
        return true;
    }

    bool accept(int component) {
        if (size[component] != int(active.size())) return false;
        if (size[component] >= bestLength) return true;
        std::string candidate(p.n, '.');
        for (int i : active) candidate[i] = Glyph[p.singleton[domain[i]]];
        return record(std::move(candidate));
    }

    bool probePearls(Clock::time_point stop) {
        std::vector<int> visited(p.n), visits(p.n), support(p.n), touched;
        touched.reserve(p.n);
        int probe = 0;
        for (int i : p.pearls) {
            if (Clock::now() >= stop || !best.empty()) break;
            const int initial = domain[i];
            if (p.count[initial] == 1) continue;
            const size_t mark = trail.size();
            int survivors = 0, directions = 0;
            touched.clear();
            for (int s = 1; s < States; ++s) {
                if (!(initial & (1 << s))) continue;
                ++probe;
                if (narrow(i, 1 << s) && propagate()) {
                    const int closed = components();
                    if (closed < 0 || accept(closed)) {
                        ++survivors;
                        directions |= 1 << s;
                        for (size_t k = mark; k < trail.size(); ++k) {
                            const int j = trail[k].cell;
                            if (visited[j] == probe) continue;
                            visited[j] = probe;
                            if (!visits[j]++) touched.push_back(j);
                            support[j] |= domain[j];
                        }
                    }
                }
                restore(mark);
            }
            if (!narrow(i, directions)) return false;
            // A cell untouched in even one survivor still has its whole prior domain.
            for (int j : touched) {
                if (visits[j] == survivors && !narrow(j, support[j])) return false;
                visits[j] = support[j] = 0;
            }
            if (!propagate()) return false;
        }
        return true;
    }

    // Frontier DP keeps the paired ends of paths in the processed region.
    // Each plug also stores its source cell's geometry and the destination requirement.
    bool narrowBoard() {
        const bool transpose = p.rows < p.cols;
        const int width = std::min(p.rows, p.cols), height = std::max(p.rows, p.cols);
        if (width > 7 || p.n < 72) return false;
        const auto stop = Clock::now() + std::chrono::milliseconds(width == 7 ? 400 : 650);
        using Key = uint64_t;
        constexpr Key Closed = Key(1) << 63;
        struct Entry { int length, history; };
        struct History { int previous, cell; Mask state; };
        std::unordered_map<Key, Entry> current, next;
        std::vector<History> history;
        current.emplace(0, Entry{0, -1});
        auto originalCell = [&](int r, int c) { return transpose ? c * p.cols + r : r * p.cols + c; };
        constexpr std::array<int, 4> transposedDirection{3, 2, 1, 0};
        std::array<int, States> originalState{};
        for (int s = 0; s < States; ++s) {
            int edges = 0;
            for (int d = 0; d < 4; ++d)
                if (Edges[s] & (1 << d)) edges |= 1 << transposedDirection[d];
            originalState[s] = transpose ? int(std::find(Edges.begin(), Edges.end(), edges) - Edges.begin()) : s;
        }
        int finalRequired = 0;
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c)
                if (!(domain[originalCell(r, c)] & Empty)) finalRequired = r * width + c;
        for (int k = 0; k < p.n; ++k) {
            if (Clock::now() >= stop || history.size() > 3500000 ||
                    current.size() > 100000) return false;
            const int c = k % width, cell = originalCell(k / width, c);
            const char clue = p.clues[cell];
            next.clear();
            next.reserve(current.size() * 2);
            for (const auto& item : current) {
                const Key key = item.first;
                const Entry entry = item.second;
                auto insert = [&](Key destination, int state) {
                    const int length = entry.length + (state != 0);
                    auto existing = next.find(destination);
                    if (existing != next.end() && existing->second.length <= length) return;
                    int trace = entry.history;
                    if (state) {
                        trace = int(history.size());
                        history.push_back({entry.history, cell, Mask(originalState[state])});
                    }
                    if (existing == next.end()) next.emplace(destination, Entry{length, trace});
                    else existing->second = {length, trace};
                };
                if (key == Closed) {
                    if (domain[cell] & Empty) insert(Closed, 0);
                    continue;
                }
                std::array<int, 8> plug{};
                for (int j = 0; j <= width; ++j) plug[j] = int((key >> (6 * j)) & 63);
                const int up = plug[c], left = plug[width];
                const int input = (up ? 1 : 0) | (left ? 8 : 0);
                if (!input && (domain[cell] & Empty)) insert(key, 0);
                for (int s = 1; s < States; ++s) {
                    if (!(domain[cell] & (1 << originalState[s])) || (Edges[s] & 9) != input)
                        continue;
                    const int out = Edges[s] & 6;
                    const int geometry = s >= 3 ? 1 : 2; // 1 requests a turn, 2 a straight.
                    if ((up >> 4 && up >> 4 != geometry) ||
                        (left >> 4 && left >> 4 != geometry)) continue;
                    const bool previousTurn = (up | left) & 8;
                    if (clue == 'B' && previousTurn) continue;
                    if (clue == 'W' && !previousTurn && !out) continue;
                    auto revised = plug;
                    revised[c] = revised[width] = 0;
                    const int a = up & 7, b = left & 7;
                    if (a && b && a == b) {
                        if (k >= finalRequired && std::none_of(revised.begin(), revised.end(),
                                                               [](int v) { return v != 0; }))
                            insert(Closed, s);
                        continue;
                    }
                    if (a && b) {
                        for (int& v : revised)
                            if ((v & 7) == b) v = (v & ~7) | a;
                    }
                    int label = a | b;
                    if (!label) {
                        // With two free slots, at most three existing pairs occupy this frontier.
                        label = 5;
                    }
                    const int metadata = label | (geometry == 1 ? 8 : 0) | (clue == 'B' ? 32 : 0);
                    if (out & 4) revised[c] = metadata;
                    if (out & 2) revised[width] = metadata;
                    const bool requireTurn = clue == 'W' && !previousTurn;
                    for (int request : {4, 2}) {
                        if (requireTurn && !(out & request)) continue;
                        if (!requireTurn && request == 2) break;
                        if (requireTurn) revised[request == 4 ? c : width] |= 16;
                        std::array<int, 8> canonical{};
                        int labels = 0;
                        Key destination = 0;
                        for (int j = 0; j <= width; ++j) {
                            const int value = revised[j], id = value & 7;
                            if (!id) continue;
                            if (!canonical[id]) canonical[id] = ++labels;
                            destination |= Key((value & ~7) | canonical[id]) << (6 * j);
                        }
                        insert(destination, s);
                        if (requireTurn) revised[request == 4 ? c : width] &= ~16;
                    }
                }
            }
            current.swap(next);
        }
        const auto it = current.find(Closed);
        if (it == current.end()) return false;
        std::string grid(p.n, '.');
        for (int trace = it->second.history; trace >= 0; trace = history[trace].previous) {
            const auto& step = history[trace];
            grid[step.cell] = Glyph[step.state];
        }
        return record(std::move(grid));
    }

    bool validPearl(const std::vector<int>& links, int i) const {
        if (p.clues[i] == '.') return true;
        const bool white = p.clues[i] == 'W';
        const int edges = links[i];
        const bool straight = edges == 5 || edges == 10;
        if (!edges || white != straight) return false;
        int straights = 0;
        for (int d = 0; d < 4; ++d) {
            if (!(edges & (1 << d))) continue;
            const int neighbor = links[p.adj[i][d]];
            straights += neighbor == 5 || neighbor == 10;
        }
        return white ? straights < 2 : straights == 2;
    }

    template<class Iterator>
    bool validatePatch(const std::vector<int>& links, Iterator first, Iterator last) const {
        for (; first != last; ++first) {
            if (!validPearl(links, *first)) return false;
            for (int j : p.adj[*first])
                if (j >= 0 && !validPearl(links, j)) return false;
        }
        return true;
    }

    bool recordLinks(const std::vector<int>& links) {
        std::string grid(p.n, '.');
        for (int i = 0; i < p.n; ++i) {
            const auto state = std::find(Edges.begin(), Edges.end(), links[i]);
            if (state == Edges.end()) return false;
            grid[i] = Glyph[state - Edges.begin()];
        }
        return record(std::move(grid));
    }

    // Join two cycles by exchanging parallel edges for the sides of an empty strip.
    bool stitch(const std::vector<int>& state, const std::vector<int>& component, int cycles) {
        std::vector<int> links(p.n), representative(p.n);
        std::iota(representative.begin(), representative.end(), 0);
        for (int i = 0; i < p.n; ++i)
            if (component[i] >= 0) links[i] = Edges[state[i]];
        auto leader = [&](int i) {
            while (representative[i] != i) {
                representative[i] = representative[representative[i]];
                i = representative[i];
            }
            return i;
        };
        std::array<int, 100> changed{}, previous{};
        std::vector<int> owner = component;
        for (int gap = 1; gap <= 8 && cycles > 1; ++gap) {
            bool merged;
            do {
                merged = false;
                for (int i = 0; i < p.n && cycles > 1; ++i) {
                    if (!links[i]) continue;
                    for (int d : {1, 2}) {
                        if (!(links[i] & (1 << d))) continue;
                        const int across = 3 - d, j = p.adj[i][d];
                        int a = i, b = j;
                        bool clear = true;
                        int count = 2;
                        changed[0] = i;
                        changed[1] = j;
                        for (int k = 1; k <= gap; ++k) {
                            a = p.adj[a][across];
                            b = p.adj[b][across];
                            if (a < 0 || b < 0 || (k < gap && (links[a] || links[b]))) {
                                clear = false;
                                break;
                            }
                            changed[count++] = a;
                            changed[count++] = b;
                        }
                        if (!clear || !(links[a] & (1 << d)) || leader(owner[i]) == leader(owner[a]))
                            continue;
                        for (int k = 0; k < count; ++k) previous[k] = links[changed[k]];
                        links[i] ^= (1 << d) | (1 << across);
                        links[j] ^= (1 << opposite(d)) | (1 << across);
                        links[a] ^= (1 << d) | (1 << opposite(across));
                        links[b] ^= (1 << opposite(d)) | (1 << opposite(across));
                        for (int k = 2; k + 2 < count; ++k)
                            links[changed[k]] = (1 << across) | (1 << opposite(across));
                        if (!validatePatch(links, changed.begin(), changed.begin() + count)) {
                            for (int k = 0; k < count; ++k) links[changed[k]] = previous[k];
                            continue;
                        }
                        representative[leader(owner[a])] = leader(owner[i]);
                        for (int k = 2; k + 2 < count; ++k) owner[changed[k]] = owner[i];
                        --cycles;
                        merged = true;
                    }
                }
            } while (merged && cycles > 1);
        }
        if (cycles > 1) return false;
        return recordLinks(links);
    }

    bool booleanFeasibility(Clock::time_point stop, bool faceParity) {
        std::vector<std::array<int, 4>> edge(p.n);
        for (auto& row : edge) row.fill(-1);
        int variables = 0;
        for (int i = 0; i < p.n; ++i)
            for (int d : {1, 2})
                if (p.possible[domain[i]] & (1 << d))
                    edge[i][d] = edge[p.adj[i][d]][opposite(d)] = variables++;
        const int firstFace = variables;
        if (faceParity) variables += (p.rows - 1) * (p.cols - 1);
        BooleanSearch sat(variables);
        auto add = [&](std::initializer_list<int> lits) { sat.constrain(std::vector<int>(lits)); };
        auto face = [&](int r, int c) {
            if (r < 0 || c < 0 || r + 1 >= p.rows || c + 1 >= p.cols) return -1;
            return 2 * (firstFace + r * (p.cols - 1) + c);
        };
        for (int i = 0; i < p.n; ++i) {
            const int available = p.possible[domain[i]];
            for (int mask = 0; mask < 16; ++mask) {
                if (mask & ~available) continue;
                const int state = int(std::find(Edges.begin(), Edges.end(), mask) - Edges.begin());
                if (state < States && (domain[i] & (1 << state))) continue;
                std::vector<int> clause;
                for (int d = 0; d < 4; ++d)
                    if (edge[i][d] >= 0) clause.push_back(2 * edge[i][d] + ((mask >> d) & 1));
                sat.constrain(std::move(clause));
            }
            for (int d : {1, 2}) {
                const int v = edge[i][d], j = p.adj[i][d];
                if (v < 0) continue;
                sat.prefer(v, 0.25 + 0.35 * ((p.clues[i] != '.') + (p.clues[j] != '.')) +
                                  double(rng() % 997) / 20000);
            }
            if (p.clues[i] == 'B') {
                for (int d = 0; d < 4; ++d) {
                    const int v = edge[i][d];
                    if (v < 0) continue;
                    const int continuation = edge[p.adj[i][d]][d];
                    if (continuation < 0) add({2 * v + 1});
                    else add({2 * v + 1, 2 * continuation});
                }
            } else if (p.clues[i] == 'W') {
                for (int d : {0, 1}) {
                    const int reverse = opposite(d), v = edge[i][d];
                    if (v < 0 || edge[i][reverse] < 0) continue;
                    const int a = edge[p.adj[i][d]][d], b = edge[p.adj[i][reverse]][reverse];
                    if (a >= 0 && b >= 0) add({2 * v + 1, 2 * a + 1, 2 * b + 1});
                }
            }
            if (!faceParity) continue;
            for (int d : {1, 2}) {
                if (p.adj[i][d] < 0) continue;
                const int r = i / p.cols, c = i % p.cols;
                const int x = face(r, c), y = d == 1 ? face(r - 1, c) : face(r, c - 1);
                const int e = 2 * edge[i][d];
                if (x >= 0 && y >= 0) {
                    if (e < 0) {
                        add({x, y ^ 1});
                        add({x ^ 1, y});
                    } else {
                        add({x, y, e ^ 1});
                        add({x ^ 1, y ^ 1, e ^ 1});
                        add({x ^ 1, y, e});
                        add({x, y ^ 1, e});
                    }
                } else {
                    const int f = std::max(x, y);
                    if (f < 0) continue;
                    if (e < 0) add({f ^ 1});
                    else {
                        add({f, e ^ 1});
                        add({f ^ 1, e});
                    }
                }
            }
        }
        // Every line separating pearls must be crossed by the final cycle.
        for (int d : {1, 2}) {
            const int lines = d == 1 ? p.cols : p.rows;
            int first = lines, last = -1;
            for (int i : p.pearls) {
                const int coordinate = d == 1 ? i % p.cols : i / p.cols;
                first = std::min(first, coordinate);
                last = std::max(last, coordinate);
            }
            for (int line = first; line < last; ++line) {
                std::vector<int> crossing;
                for (int i = 0; i < p.n; ++i) {
                    const int coordinate = d == 1 ? i % p.cols : i / p.cols;
                    if (coordinate == line && edge[i][d] >= 0) crossing.push_back(2 * edge[i][d]);
                }
                sat.constrain(std::move(crossing));
            }
        }
        sat.initializeOrder();
        std::vector<int> component(p.n), state(p.n), walk;
        walk.reserve(p.n);
        std::vector<int> reached(p.n), reachable;
        reachable.reserve(p.n);
        int generation = 0;
        auto disconnectedCut = [&]() {
            ++generation;
            reachable.clear();
            const int first = p.pearls.front();
            reachable.push_back(first);
            reached[first] = generation;
            int missing = int(p.pearls.size());
            for (size_t k = 0; k < reachable.size(); ++k) {
                const int i = reachable[k];
                if (p.clues[i] != '.' && !--missing) return std::vector<int>{};
                for (int d = 0; d < 4; ++d) {
                    const int variable = edge[i][d];
                    if (variable < 0 || sat.off(variable)) continue;
                    const int j = p.adj[i][d];
                    if (reached[j] == generation) continue;
                    reached[j] = generation;
                    reachable.push_back(j);
                }
            }
            std::vector<int> cut;
            for (int i : reachable)
                for (int d = 0; d < 4; ++d)
                    if (edge[i][d] >= 0 && reached[p.adj[i][d]] != generation)
                        cut.push_back(2 * edge[i][d]);
            return cut;
        };
        std::vector<int> dfs, dfsOrder, dfsIndex(p.n), dfsLow(p.n), dfsParent(p.n),
            dfsNext(p.n), dfsPearls(p.n);
        dfs.reserve(p.n);
        dfsOrder.reserve(p.n);
        auto separatorCut = [&]() -> std::vector<int> {
            ++generation;
            dfs.clear();
            dfsOrder.clear();
            auto visit = [&](int i, int from) {
                reached[i] = generation;
                dfsIndex[i] = dfsLow[i] = int(dfsOrder.size());
                dfsParent[i] = from;
                dfsNext[i] = 0;
                dfsPearls[i] = p.clues[i] != '.';
                dfsOrder.push_back(i);
                dfs.push_back(i);
            };
            visit(p.pearls.front(), -1);
            const int total = int(p.pearls.size());
            while (!dfs.empty()) {
                const int i = dfs.back();
                if (dfsNext[i] < 4) {
                    const int d = dfsNext[i]++, v = edge[i][d];
                    if (v < 0 || sat.off(v)) continue;
                    const int j = p.adj[i][d];
                    if (reached[j] != generation) visit(j, i);
                    else if (j != dfsParent[i]) dfsLow[i] = std::min(dfsLow[i], dfsIndex[j]);
                    continue;
                }
                dfs.pop_back();
                const int parentCell = dfsParent[i];
                if (parentCell >= 0) {
                    dfsLow[parentCell] = std::min(dfsLow[parentCell], dfsLow[i]);
                    dfsPearls[parentCell] += dfsPearls[i];
                }
                if (!dfsPearls[i] || dfsPearls[i] == total) continue;
                const bool articulation = parentCell >= 0 && dfsLow[i] >= dfsIndex[parentCell] &&
                    total - dfsPearls[i] > (p.clues[parentCell] != '.');
                const bool bridge = parentCell >= 0 && dfsLow[i] > dfsIndex[parentCell];
                if (parentCell >= 0 && !articulation && !bridge) continue;
                // Removing one articulation vertex or leaving one crossing edge
                // cannot join pearls on opposite shores of a simple cycle.
                const int first = dfsIndex[i], last = int(dfsOrder.size());
                std::vector<int> cut;
                for (int k = first; k < last; ++k) {
                    const int cell = dfsOrder[k];
                    for (int d = 0; d < 4; ++d) {
                        if (edge[cell][d] < 0) continue;
                        const int j = p.adj[cell][d];
                        if (reached[j] == generation && dfsIndex[j] >= first) continue;
                        if (j == parentCell && (articulation || cell == i)) continue;
                        cut.push_back(2 * edge[cell][d]);
                    }
                }
                if (!cut.empty()) return cut;
            }
            return {};
        };
        unsigned checks = 0;
        auto connectivity = [&]() {
            return (++checks % (faceParity ? 2 : 3)) == 0 ? separatorCut() : disconnectedCut();
        };
        while (Clock::now() < stop && sat.solve(stop, connectivity, p.n < 400 ? 13 : 23)) {
#ifdef MASYU_STATS
            ++satModels;
#endif
            std::fill(component.begin(), component.end(), -1);
            for (int i = 0; i < p.n; ++i) {
                int links = 0;
                for (int d = 0; d < 4; ++d)
                    if (edge[i][d] >= 0 && sat.on(edge[i][d])) links |= 1 << d;
                state[i] = int(std::find(Edges.begin(), Edges.end(), links) - Edges.begin());
                if (state[i] == States) return false;
            }
            std::vector<std::vector<int>> cuts;
            for (int pearl : p.pearls) {
                if (component[pearl] >= 0) continue;
                walk.clear();
                walk.push_back(pearl);
                component[pearl] = pearl;
                int pearlsHere = 0;
                for (size_t k = 0; k < walk.size(); ++k) {
                    const int i = walk[k];
                    pearlsHere += p.clues[i] != '.';
                    for (int d = 0; d < 4; ++d) {
                        if (!(Edges[state[i]] & (1 << d))) continue;
                        const int j = p.adj[i][d];
                        if (component[j] >= 0) continue;
                        component[j] = pearl;
                        walk.push_back(j);
                    }
                }
                if (pearlsHere == int(p.pearls.size())) {
                    std::string grid(p.n, '.');
                    for (int i : walk) grid[i] = Glyph[state[i]];
                    return record(std::move(grid));
                }
                std::vector<int> boundary, brokenLoop;
                for (int i : walk) {
                    for (int d = 0; d < 4; ++d) {
                        if (edge[i][d] < 0) continue;
                        const int j = p.adj[i][d];
                        if (component[j] != pearl) boundary.push_back(2 * edge[i][d]);
                        if ((d == 1 || d == 2) && (Edges[state[i]] & (1 << d)))
                            brokenLoop.push_back(2 * edge[i][d] + 1);
                    }
                }
                cuts.push_back(std::move(boundary));
                cuts.push_back(std::move(brokenLoop));
            }
            if (stitch(state, component, int(cuts.size() / 2))) return true;
            sat.reset();
            for (auto& cut : cuts) sat.constrain(std::move(cut));
        }
        return false;
    }

    struct Decision {
        int cell = -1;
        std::array<Mask, States> choices{};
        int count = 0;
    };

    Decision choose() {
        int cell = -1, priority = -Inf;
        auto nearby = [&](int i) {
            int value = 0;
            for (int j : p.adj[i]) {
                if (j < 0) continue;
                value += (p.clues[j] != '.') * 5;
                value += !(domain[j] & Empty) ? 3 : 0;
            }
            return value;
        };
        for (int i : p.pearls) {
            const int count = p.count[domain[i]];
            if (count == 1) continue;
            int score = -16 * count + nearby(i);
            if (mode) score += int(rng() % (mode == 1 ? 17 : 35));
            if (score > priority) {
                priority = score;
                cell = i;
            }
        }
        const bool selectingPearl = cell >= 0;
        // Grow the most constrained path end, unless a pearl orientation remains undecided.
        if (!selectingPearl) {
            for (int i : endpoints) {
                const int count = p.count[domain[i]];
                if (count == 1) continue;
                int score = -16 * count + nearby(i) + (degree[i] == 0 ? 12 : 0);
                if (mode) score += int(rng() % (mode == 1 ? 17 : 35));
                if (score > priority) {
                    priority = score;
                    cell = i;
                }
            }
        }
        Decision decision;
        if (cell < 0) return decision;
        decision.cell = cell;
        const int component = root(cell);
        const bool canClose = size[component] == int(active.size());
        if (!selectingPearl) {
            std::fill(distance.begin(), distance.end(), Inf);
            bfs.clear();
            for (int i : endpoints) {
                if (i == cell || (!canClose && root(i) == component)) continue;
                distance[i] = 0;
                bfs.push_back(i);
            }
            int targets = 0;
            std::array<int, 4> targetCells{-1, -1, -1, -1};
            const int exits = p.possible[domain[cell]] & ~p.forced[domain[cell]];
            for (int d = 0; d < 4; ++d) {
                if (!(exits & (1 << d))) continue;
                const int j = p.adj[cell][d];
                if (domain[j] & Empty) targetCells[targets++] = j;
            }
            for (size_t k = 0; k < bfs.size() && targets; ++k) {
                const int i = bfs[k];
                const int edges = p.possible[domain[i]] & ~p.forced[domain[i]];
                for (int d = 0; d < 4; ++d) {
                    if (!(edges & (1 << d))) continue;
                    const int j = p.adj[i][d];
                    if (distance[j] != Inf || !(domain[j] & Empty)) continue;
                    distance[j] = distance[i] + 1;
                    bfs.push_back(j);
                    const auto end = targetCells.begin() + targets;
                    const auto target = std::find(targetCells.begin(), end, j);
                    if (target != end) *target = targetCells[--targets];
                }
            }
        }

        std::array<std::pair<int, int>, States> order{};
        const int existing = p.forced[domain[cell]];
        for (int s = 1; s < States; ++s) {
            if (!(domain[cell] & (1 << s))) continue;
            int cost = 0;
            int newCells = 0;
            bool valid = true;
            for (int d = 0; d < 4; ++d) {
                if (!(Edges[s] & ~existing & (1 << d))) continue;
                const int j = p.adj[cell][d];
                const bool mandatory = !(domain[j] & Empty);
                if (mandatory && !canClose && root(j) == component) {
                    valid = false;
                    break;
                }
                if (selectingPearl) {
                    cost += mandatory ? -9 : 4;
                    for (int k : p.adj[j])
                        if (k >= 0 && p.clues[k] != '.') cost -= 2;
                } else {
                    if (distance[j] == Inf) { valid = false; break; }
                    newCells += distance[j];
                    cost += 4 * distance[j] + (mandatory ? 0 : 2);
                }
            }
            if (!valid || required[0] + required[1] + newCells >= bestLength) continue;
            if (mode) cost += int(rng() % (mode == 1 ? 7 : 17));
            order[decision.count++] = {cost, 1 << s};
        }
        for (int k = 1; k < decision.count; ++k) {
            const auto choice = order[k];
            int j = k;
            while (j > 0 && choice < order[j - 1]) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = choice;
        }
        for (int k = 0; k < decision.count; ++k) decision.choices[k] = Mask(order[k].second);
        return decision;
    }

public:
    std::string best;
    int bestLength = Inf, improvements = 0;
    uint64_t nodes = 0, propagations = 0, contradictions = 0, connectivityPrunes = 0, satModels = 0;
    double firstTime = 0;

    explicit Solver(const Puzzle& puzzle)
        : p(puzzle), domain(p.initial), queued(p.n), parent(p.n), size(p.n), degree(p.n),
          distance(p.n), discovery(p.n), low(p.n), predecessor(p.n), cursor(p.n), kept(p.n) {
        for (int i = 0; i < p.n; ++i) {
            if (!(domain[i] & Empty)) ++required[color(i)];
            enqueue(i);
        }
        trail.reserve(p.n * 6);
        active.reserve(p.n);
        endpoints.reserve(p.n);
        bfs.reserve(p.n);
        vertexStack.reserve(p.n);
        edgeStack.reserve(p.n * 2);
        int top = p.rows, bottom = 0, left = p.cols, right = 0;
        for (int i : p.pearls) {
            top = std::min(top, i / p.cols);
            bottom = std::max(bottom, i / p.cols);
            left = std::min(left, i % p.cols);
            right = std::max(right, i % p.cols);
        }
        geometricBound = 2 * (bottom - top + right - left);
        // Shortcutting the loop between pearls gives a metric tour. A spanning
        // tree plus the cheapest distinct-pearl edge is a lower bound on that tour.
        const int k = int(p.pearls.size());
        std::vector<int> nearest(k, Inf);
        nearest[0] = 0;
        int spanning = 0, shortest = Inf;
        for (int step = 0; step < k; ++step) {
            const int i = int(std::min_element(nearest.begin(), nearest.end()) - nearest.begin());
            spanning += nearest[i];
            nearest[i] = Inf + 1;
            for (int j = 0; j < k; ++j) {
                if (nearest[j] > Inf) continue;
                const int d = manhattan(p.pearls[i], p.pearls[j]);
                shortest = std::min(shortest, d);
                nearest[j] = std::min(nearest[j], d);
            }
        }
        geometricBound = std::max(geometricBound, (spanning + shortest + 1) & ~1);
    }

    enum Result { Exhausted, Budget, Improved };
    Result search(uint64_t limit, int strategy, bool stopOnImprovement) {
        struct Frame { size_t mark; Decision branch; int next; };
        std::vector<Frame> stack;
        stack.reserve(p.n);
        mode = strategy;
        const size_t base = trail.size();
        const uint64_t end = nodes + limit;
        const int originalBest = bestLength;
        bool consistent = propagate();
        while (true) {
            ++nodes;
            if (nodes > end || ((nodes & 7) == 0 && Clock::now() >= deadline)) {
                restore(base);
                return Budget;
            }
            if (consistent && std::max(geometricBound, 2 * std::max(required[0], required[1])) >= bestLength)
                consistent = false;
            if (consistent && (stack.empty() || (nodes & 15) == 0) && !restrictBlock()) {
                ++connectivityPrunes;
                consistent = false;
            }
            if (consistent) {
                const int closed = components();
                if (closed >= 0) {
                    if (!accept(closed)) ++connectivityPrunes;
                    if (stopOnImprovement && bestLength < originalBest) {
                        restore(base);
                        return Improved;
                    }
                } else {
                    Decision branch = choose();
                    if (branch.count) stack.push_back({trail.size(), branch, 0});
                }
            } else {
                ++contradictions;
            }
            while (!stack.empty() && stack.back().next == stack.back().branch.count) {
                restore(stack.back().mark);
                stack.pop_back();
            }
            if (stack.empty()) {
                restore(base);
                return Exhausted;
            }
            Frame& frame = stack.back();
            restore(frame.mark);
            consistent = narrow(frame.branch.cell, frame.branch.choices[frame.next++]) && propagate();
        }
    }

    void setDeadline(double seconds) {
        deadline = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    }

    std::vector<int> loopOrder() const {
        std::vector<int> loop;
        int cell = int(best.find_first_not_of('.')), previous = -1;
        do {
            loop.push_back(cell);
            int state = int(std::find(Glyph, Glyph + States, best[cell]) - Glyph);
            for (int d = 0; d < 4; ++d) {
                if (!(Edges[state] & (1 << d)) || p.adj[cell][d] == previous) continue;
                previous = cell;
                cell = p.adj[cell][d];
                break;
            }
        } while (cell != loop.front());
        return loop;
    }

    void shortcut() {
        const auto loop = loopOrder();
        std::vector<int> next(p.n, -1), links(p.n);
        for (size_t k = 0; k < loop.size(); ++k) {
            const int i = loop[k];
            next[i] = loop[(k + 1) % loop.size()];
            links[i] = Edges[std::find(Glyph, Glyph + States, best[i]) - Glyph];
        }
        auto edgeBit = [&](int i, int j) {
            return 1 << (std::find(p.adj[i].begin(), p.adj[i].end(), j) - p.adj[i].begin());
        };
        std::vector<int> removed, newPath, changed, previous;
        bool improved = false;
        for (int pass = 0; pass < 3; ++pass) {
            bool edited = false;
            for (int a = 0; a < p.n; ++a) {
                if (next[a] < 0) continue;
                removed.clear();
                int b = next[a];
                for (int steps = 1; steps <= 32 && b != a; ++steps) {
                    const int r = b / p.cols - a / p.cols, c = b % p.cols - a % p.cols;
                    if (steps >= 3 && (!r || !c) && std::abs(r + c) < steps) {
                        const int d = r ? (r > 0 ? 2 : 0) : (c > 0 ? 1 : 3);
                        const int back = opposite(d);
                        if (!(links[a] & (1 << d))) {
                            newPath.clear();
                            int i = p.adj[a][d];
                            while (i != b && !links[i]) {
                                newPath.push_back(i);
                                i = p.adj[i][d];
                            }
                            if (i == b) {
                                changed = removed;
                                changed.insert(changed.end(), newPath.begin(), newPath.end());
                                changed.push_back(a);
                                changed.push_back(b);
                                previous.clear();
                                for (int j : changed) previous.push_back(links[j]);
                                links[a] ^= edgeBit(a, next[a]) | (1 << d);
                                links[b] ^= edgeBit(b, removed.back()) | (1 << back);
                                for (int j : removed) links[j] = 0;
                                for (int j : newPath) links[j] = (1 << d) | (1 << back);
                                if (validatePatch(links, changed.begin(), changed.end())) {
                                    for (int j : removed) next[j] = -1;
                                    i = a;
                                    for (int j : newPath) {
                                        next[i] = j;
                                        i = j;
                                    }
                                    next[i] = b;
                                    edited = true;
                                    break;
                                }
                                for (size_t k = 0; k < changed.size(); ++k)
                                    links[changed[k]] = previous[k];
                            }
                        }
                    }
                    if (p.clues[b] != '.') break;
                    removed.push_back(b);
                    b = next[b];
                }
            }
            improved |= edited;
            if (!edited) break;
        }
        if (improved) recordLinks(links);
    }

    // Reroute a pearl-free arc between fixed ends using shortest-path search
    // over (cell, incoming direction). The rest of the loop is an obstacle.
    void rerouteArcs(Clock::time_point stop) {
        struct Arc { int offset, edges, saving; };
        std::vector<int> occupied(p.n), previous(4 * p.n), depth(4 * p.n), work;
        work.reserve(4 * p.n);
        int attempts = 0;
        while (Clock::now() < stop && attempts < 120) {
            const auto loop = loopOrder();
            const int length = int(loop.size());
            auto at = [&](int k) { return loop[(k + length) % length]; };
            std::vector<Arc> arcs;
            for (int i = 0; i < length; ++i) occupied[loop[i]] = 1;
            for (int i = 0; i < length; ++i) {
                if (p.clues[at(i - 1)] == '.' && p.clues[at(i)] == '.' && i % 4) continue;
                for (int steps = 1; steps <= std::min(140, length - 2); ++steps) {
                    const int b = at(i + steps);
                    const bool pearl = p.clues[b] != '.';
                    if (steps >= 3 && (pearl || p.clues[at(i + steps + 1)] != '.' || steps % 4 == 0)) {
                        const int saving = steps - manhattan(at(i), b);
                        if (saving >= 2) arcs.push_back({i, steps, saving});
                    }
                    if (pearl) break;
                }
            }
            std::sort(arcs.begin(), arcs.end(), [](const Arc& a, const Arc& b) {
                return a.saving > b.saving;
            });
            bool improved = false;
            for (const Arc& arc : arcs) {
                if (attempts++ >= 120 || Clock::now() >= stop) break;
                const int a = at(arc.offset), b = at(arc.offset + arc.edges);
                auto direction = [&](int i, int j) {
                    return int(std::find(p.adj[i].begin(), p.adj[i].end(), j) - p.adj[i].begin());
                };
                const int beforeA = at(arc.offset - 1), afterB = at(arc.offset + arc.edges + 1);
                const int enter = direction(a, beforeA), exit = direction(b, afterB);
                auto straight = [&](int i) { return best[i] == '-' || best[i] == '|'; };
                auto requiredNeighbor = [&](int pearl, int fixedNeighbor) {
                    if (p.clues[pearl] == 'B') return Straight;
                    if (p.clues[pearl] == 'W' && straight(fixedNeighbor)) return Turn;
                    return All;
                };
                const int firstMask = requiredNeighbor(a, beforeA);
                const int lastMask = requiredNeighbor(b, afterB);
                const int startMask = requiredNeighbor(beforeA, at((arc.offset + length - 2) % length));
                const int endMask = requiredNeighbor(afterB, at(arc.offset + arc.edges + 2));
                for (int k = 1; k < arc.edges; ++k) occupied[at(arc.offset + k)] = 0;
                std::fill(previous.begin(), previous.end(), -2);
                work.clear();
                const int source = 4 * a + enter;
                previous[source] = -1;
                depth[source] = 0;
                work.push_back(source);
                int target = -1;
                for (size_t k = 0; k < work.size(); ++k) {
                    const int node = work[k], i = node / 4, in = node % 4;
                    if (i == b) {
                        if (in != exit && (domain[b] & endMask & p.uses[in] & p.uses[exit])) {
                            target = node;
                            break;
                        }
                        continue;
                    }
                    int mask = domain[i] & p.uses[in];
                    if (node == source) mask &= startMask;
                    if (depth[node] == 1) mask &= firstMask;
                    const int allowedEdges = p.possible[mask];
                    for (int d = 0; d < 4; ++d) {
                        if (d == in || !(allowedEdges & (1 << d))) continue;
                        const int j = p.adj[i][d];
                        if ((occupied[j] && j != b) || depth[node] + 1 + manhattan(j, b) >= arc.edges)
                            continue;
                        if (j == b && !(mask & lastMask & p.uses[d])) continue;
                        if (node == source && j == b &&
                            !(domain[b] & firstMask & endMask & p.uses[opposite(d)] & p.uses[exit]))
                            continue;
                        const int next = 4 * j + opposite(d);
                        if (previous[next] != -2) continue;
                        previous[next] = node;
                        depth[next] = depth[node] + 1;
                        work.push_back(next);
                    }
                }
                for (int k = 1; k < arc.edges; ++k) occupied[at(arc.offset + k)] = 1;
                if (target < 0) continue;
                std::vector<int> links(p.n);
                for (int i : loop) links[i] = Edges[std::find(Glyph, Glyph + States, best[i]) - Glyph];
                for (int k = 1; k < arc.edges; ++k) links[at(arc.offset + k)] = 0;
                links[b] = (1 << exit) | (1 << (target % 4));
                for (int node = target; previous[node] >= 0; node = previous[node]) {
                    const int predecessor = previous[node];
                    links[predecessor / 4] = (1 << (predecessor % 4)) | (1 << opposite(node % 4));
                }
                const int before = bestLength;
                recordLinks(links);
                if (bestLength < before) {
                    improved = true;
                    break;
                }
            }
            if (!improved) break;
            std::fill(occupied.begin(), occupied.end(), 0);
        }
    }

    void optimize(double stop, double hardStop) {
        setDeadline(stop);
        if (bestLength > geometricBound && elapsed() + 0.015 < stop) shortcut();
        if (p.n >= 600 && bestLength > geometricBound && elapsed() + 0.025 < stop)
            rerouteArcs(std::min(deadline, Clock::now() + std::chrono::milliseconds(115)));
        if (p.n <= 100) {
            if (search(12000, 0, false) == Exhausted) return;
        }
        int attempt = 0, unsuccessful = 0;
        double checkpoint = firstTime;
        int checkpointLength = bestLength;
        auto speed = [](double t) { return 0.5 + 0.5 * std::exp(-std::max(0.0, t - 1.0) / 5.0); };
        while (elapsed() < stop && unsuccessful < 100 && bestLength > geometricBound) {
            const auto loop = loopOrder();
            const int length = int(loop.size());
            const int width = std::min(length - 2, 5 + int(rng() % (attempt < 30 ? 14 : 45)));
            int offset = int(rng() % length);
            if (attempt % 3 != 0) {
                auto detour = [&](int first) {
                    int pearls = 0;
                    for (int k = 0; k < width; ++k)
                        pearls += p.clues[loop[(first + k) % length]] != '.';
                    const int a = loop[(first + length - 1) % length], b = loop[(first + width) % length];
                    return 3 * (width + 1 - manhattan(a, b)) - pearls;
                };
                int largest = detour(offset);
                for (int k = 0; k < 7; ++k) {
                    const int candidate = int(rng() % length), savings = detour(candidate);
                    if (savings <= largest) continue;
                    largest = savings;
                    offset = candidate;
                }
            }
            const int previousBest = bestLength;
            const size_t mark = trail.size();
            bool consistent = true;
            const bool spatial = p.n >= 600 && attempt % 4 == 2;
            ++attempt;
            if (spatial) {
                const int center = loop[(offset + width / 2) % length];
                const int radius = 3 + int(rng() % 7);
                for (int i : loop) {
                    if (manhattan(i, center) <= radius) continue;
                    const int state = int(std::find(Glyph, Glyph + States, best[i]) - Glyph);
                    if (!narrow(i, 1 << state)) { consistent = false; break; }
                }
            } else if (attempt % 9 != 0) {
                for (int k = width; k < length; ++k) {
                    const int i = loop[(offset + k) % length];
                    const int state = int(std::find(Glyph, Glyph + States, best[i]) - Glyph);
                    if (!narrow(i, 1 << state)) { consistent = false; break; }
                }
            }
            if (consistent) search(spatial ? 110 : attempt % 9 ? 160 : 1000, attempt % 3, true);
            restore(mark);
            unsuccessful = bestLength < previousBest ? 0 : unsuccessful + 1;
            const double now = elapsed();
            if (now >= stop && firstTime >= 0.9 && stop < std::min(hardStop, firstTime + 0.8)) {
                const double ratio = double(checkpointLength) / bestLength;
                const double cost = speed(checkpoint) / speed(now);
                if (ratio * ratio > std::max(1.012, cost) && 10 * bestLength > 11 * geometricBound) {
                    stop = std::min({now + 0.16, hardStop, firstTime + 0.8});
                    setDeadline(stop);
                    checkpoint = now;
                    checkpointLength = bestLength;
                }
            }
        }
        if (bestLength > geometricBound && elapsed() + 0.015 < hardStop) shortcut();
    }

    void tryCorridor() {
        if (p.n < 600 || p.pearls.size() < 12) return;
        constexpr int radius = 5;
        std::fill(distance.begin(), distance.end(), Inf);
        bfs = p.pearls;
        for (int i : bfs) distance[i] = 0;
        for (size_t k = 0; k < bfs.size(); ++k) {
            const int i = bfs[k];
            if (distance[i] == radius) continue;
            for (int j : p.adj[i]) {
                if (j < 0 || distance[j] != Inf) continue;
                distance[j] = distance[i] + 1;
                bfs.push_back(j);
            }
        }
        int removable = 0;
        for (int i = 0; i < p.n; ++i)
            removable += distance[i] == Inf && (domain[i] & Empty) && domain[i] != Empty;
        if (removable < p.n / 8) return;
        const size_t mark = trail.size();
        for (int i = 0; i < p.n; ++i)
            if (distance[i] == Inf && (domain[i] & Empty)) narrow(i, Empty);
        if (propagate() && restrictBlock()) {
            setDeadline(std::min(1.05, elapsed() + 0.75));
            booleanFeasibility(deadline, true);
        }
        restore(mark);
    }

    void solve() {
        if (!propagate() || !restrictBlock()) return;
        trail.clear();
        constexpr double hardStop = 9.55;
        const bool exact = narrowBoard();
        if (!exact && p.pearls.size() > 35) {
            if (!probePearls(Clock::now() + std::chrono::milliseconds(35)) || !restrictBlock())
                return;
            trail.clear();
        }
        setDeadline(std::min(hardStop, elapsed() + 0.18));
        for (int attempt = 0; best.empty() && attempt < 4; ++attempt) {
            if (Clock::now() >= deadline) break;
            if (search(p.n < 400 ? 800 : 350, attempt % 3, true) == Exhausted) return;
        }
        if (best.empty()) tryCorridor();
        const bool parityFirst = p.n >= 1000;
        if (best.empty()) {
            setDeadline(4.4);
            booleanFeasibility(deadline, parityFirst);
        }
        if (best.empty()) {
            setDeadline(8.55);
            booleanFeasibility(deadline, !parityFirst);
        }
        setDeadline(hardStop);
        for (int attempt = 0; best.empty() && elapsed() < hardStop; ++attempt) {
            const uint64_t scale = uint64_t(1) << std::min(attempt / 3, 12);
            const uint64_t budget = (p.n < 400 ? 1600 : 400) * scale;
            const Result result = search(budget, attempt % 3, true);
            if (result == Exhausted) return;
        }
        if (!exact && !best.empty())
            optimize(std::min(hardStop, std::max(0.94, firstTime + 0.18)), hardStop);
#ifdef MASYU_STATS
        const double total = elapsed();
        std::cerr << "total=" << total << " first=" << firstTime
                  << " optimization=" << (best.empty() ? 0.0 : total - firstTime)
                  << " length=" << (best.empty() ? 0 : bestLength) << " nodes=" << nodes
                  << " propagation=" << propagations << " contradictions=" << contradictions
                  << " connectivity_prunes=" << connectivityPrunes << " sat_models=" << satModels
                  << " improvements=" << improvements << '\n';
#endif
    }
};
}  // namespace

// Returns an empty vector if the input is invalid or no feasible loop is found.
std::vector<std::string> solve(std::vector<std::string> grid) {
    if (grid.size() < 6 || grid.size() > 50) return {};
    for (const auto& row : grid) {
        if (row.size() < 6 || row.size() > 50 ||
            row.size() != grid.front().size() || row.find_first_not_of(".WB") != std::string::npos)
            return {};
    }
    const Puzzle puzzle(grid);
    if (puzzle.pearls.size() < 2) return {};
    Solver solver(puzzle);
    solver.solve();
    if (solver.best.empty()) return {};
    for (int r = 0; r < puzzle.rows; ++r)
        grid[r].assign(solver.best, r * puzzle.cols, puzzle.cols);
    return grid;
}
