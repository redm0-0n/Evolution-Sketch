#include <opencv2/opencv.hpp>     // OpenCV: image I/O + matrices + simple drawing (line into mask)
#include <filesystem>            // std::filesystem: directories, paths, file naming
#include <random>                // RNG: random_device, mt19937, distributions
#include <string>                // std::string, std::to_string
#include <iostream>              // std::cout for console logs
#include <chrono>                // time measurement (run time limit, logging timestamps)
#include <cmath>                 // lround for numeric rounding
#include <vector>                // std::vector for populations and buffers
#include <algorithm>             // sort, min, max
#include <cstdlib>               // _putenv_s on Windows (disable OpenCV plugins)
#include <fstream>               // logs.txt writing

using namespace std;             // Simpler code style (as in your original solution)

// rd: non-deterministic seed source
random_device rd;
// rng: fast pseudo-random generator used everywhere in the algorithm
mt19937 rng(rd());

const int inputIndex = 1;        // Which input image folder and file index to use (input1.jpg, input2.jpg, etc.)
const int RUNS = 3;              // Number of independent runs. Each run must complete within time limit.

const int SIZE = 512;            // Working resolution (canvas and target are resized to SIZE x SIZE)

const int LINE_NUMBER = 10000;   // Maximum number of strokes (lines) per run
const int RUN_TIME_LIMIT_SEC = 540; // Maximum wall-clock time per run (seconds)

const int LOG_EVERY_LINES = 50;  // Log every X strokes
const int LAST_KEEP = 5;         // Keep last N snapshots per run

// These ranges define how parameters change over time (t from 0 to 1).
// Early = more exploration (bigger lines, bigger mutation).
// Late  = more exploitation (smaller lines, smaller mutation).

const int L_MIN = 4;             // Minimum line length (constant)
const int L_MAX_MIN = 16;        // Minimum allowed maximum length near the end
const int L_MAX_MAX = 48;        // Maximum allowed maximum length at the beginning

// Shade constraints: pencil vs eraser
const int PENCIL_SHADE_MAX = 220; // Pencil shade will be sampled in [0..PENCIL_SHADE_MAX]
const int ERASER_SHADE_MIN = 200; // Eraser shade will be sampled in [ERASER_SHADE_MIN..255]

// Stroke pressure ranges: used as alpha blending (0..255 mapped to 0..1)
const int PRESS_MIN_MIN = 30;
const int PRESS_MIN_MAX = 60;
const int PRESS_MAX_MIN = 110;
const int PRESS_MAX_MAX = 170;

// Mutation ranges scheduled over time
const int DELTA_XY_MIN = 4;
const int DELTA_XY_MAX = 10;

const int DELTA_SHADE_MIN = 10;
const int DELTA_SHADE_MAX = 25;

const int DELTA_PRESS_MIN = 8;
const int DELTA_PRESS_MAX = 18;

// Probability ranges scheduled over time
const double ERASE_PROB_MIN = 0.20;      // Probability that a random stroke becomes "eraser" at end
const double ERASE_PROB_MAX = 0.45;      // Probability that a random stroke becomes "eraser" at start

const double RESET_PROB_MIN = 0.01;      // Chance that mutation resets a stroke at end
const double RESET_PROB_MAX = 0.04;      // Chance that mutation resets a stroke at start

const double DELTA_TYPE_PROB_MIN = 0.01; // Chance mutation flips pencil/eraser at end
const double DELTA_TYPE_PROB_MAX = 0.03; // Chance mutation flips pencil/eraser at start

// EA inner-loop settings
const int CHILD_NUMBER = 35;             // Number of children produced per EA generation
const int GENERATIONS = 30;              // EA generations per one applied stroke

const int POPULATION_SIZE = 30;          // Population size
const int TOURNAMENT_PARTICIPANTS = 5;   // Tournament size
const int TOURNAMENT_WINNER = 8;         // How many winners to sample for parent pool

// Output folder
const filesystem::path working_dir = filesystem::path("E:\\ASSIGMENT2_VERSION2\\") / ("input" + to_string(inputIndex));



// Correction is used to keep parameters and coordinates inside valid ranges.
static int correction(int value, int lower, int upper) {
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}

// Changing int parameters over time
static int new_param_int(int a, int b, double t) {
    return (int)lround((1.0 - t) * (double)a + t * (double)b);
}

// Changing double parameters over time
static double new_param_double(double a, double b, double t) {
    return (1.0 - t) * a + t * b;
}

// This attempts to reduce OpenCV's dynamic plugin usage and parallelism.
// It does NOT create an EA; it only makes OpenCV behave "more simple".
static void disable_opencv_plugins_and_parallel() {
#ifdef _WIN32
    // Reduce OpenCV verbosity and disable plugin search path.
    _putenv_s("OPENCV_LOG_LEVEL", "ERROR");
    _putenv_s("OPENCV_CORE_PLUGIN_PATH", "");
#endif
    // Disable OpenCV internal optimizations and multithreading.
    // This makes behavior more predictable and avoids external backends.
    cv::setUseOptimized(false);
    cv::setNumThreads(1);
}

// Params wraps all "current" values used for line generation and mutation,
// and EA sizes (population / tournament / etc.).
struct Params {
    int L_min;
    int L_max;
    int press_min;
    int press_max;

    int delta_xy;
    int delta_shade;
    int delta_press;

    double erase_prob;
    double reset_prob;
    double delta_type_prob;

    int population_size;
    int tournament_participants;
    int tournament_winner;
    int child_number;
    int generations;
};

// Compute scheduled parameters for progress t
// t = 0 -> early stage: longer strokes, stronger mutations, more eraser.
// t = 1 -> late stage: shorter strokes, weaker mutations, less eraser.
static Params get_params(double t) {
    Params p{};

    // Stroke length range: max length decreases over time.
    p.L_min = L_MIN;
    p.L_max = correction(new_param_int(L_MAX_MAX, L_MAX_MIN, t), L_MAX_MIN, L_MAX_MAX);

    // Pressure range: becomes weaker over time.
    p.press_min = correction(new_param_int(PRESS_MIN_MAX, PRESS_MIN_MIN, t), PRESS_MIN_MIN, PRESS_MIN_MAX);
    p.press_max = correction(new_param_int(PRESS_MAX_MAX, PRESS_MAX_MIN, t), PRESS_MAX_MIN, PRESS_MAX_MAX);

    // Mutation magnitudes: become smaller over time.
    p.delta_xy = correction(new_param_int(DELTA_XY_MAX, DELTA_XY_MIN, t), DELTA_XY_MIN, DELTA_XY_MAX);
    p.delta_shade = correction(new_param_int(DELTA_SHADE_MAX, DELTA_SHADE_MIN, t), DELTA_SHADE_MIN, DELTA_SHADE_MAX);
    p.delta_press = correction(new_param_int(DELTA_PRESS_MAX, DELTA_PRESS_MIN, t), DELTA_PRESS_MIN, DELTA_PRESS_MAX);

    // Probability schedule:
    // eraser probability decreases so image does not get "washed out" near the end.
    p.erase_prob = new_param_double(ERASE_PROB_MAX, ERASE_PROB_MIN, t);

    // Reset and type flip probabilities decrease over time (less randomness late).
    p.reset_prob = new_param_double(RESET_PROB_MAX, RESET_PROB_MIN, t);
    p.delta_type_prob = new_param_double(DELTA_TYPE_PROB_MAX, DELTA_TYPE_PROB_MIN, t);

    // EA sizes (fixed in this solution).
    p.population_size = POPULATION_SIZE;
    p.tournament_participants = TOURNAMENT_PARTICIPANTS;
    p.tournament_winner = TOURNAMENT_WINNER;

    p.child_number = CHILD_NUMBER;
    p.generations = GENERATIONS;

    return p;
}

// Representation of one stroke (chromosome)
// A chromosome is one line from (x_start, y_start) to (x_finish, y_finish)
// with shade (0..255), press (0..255), type (0 pencil, 1 eraser).
class Line {
private:
    int x_start, y_start, x_finish, y_finish; // endpoints (canvas coordinates)
    int shade;                                // desired shade when drawing (0=black, 255=white)
    int press;                                // pressure -> alpha blending strength
    int type;                                 // 0 = pencil, 1 = eraser

public:
    // Default constructor: creates a "zero" line. Mostly used for temporary objects.
    Line() : x_start(0), y_start(0), x_finish(0), y_finish(0), shade(0), press(0), type(0) {}



    // Random line initialization using current Params.
    // Generation is purely random, not directed by the target image.
    Line(const Params& p) {
        // Random distributions for coordinates and parameters.
        uniform_int_distribution<int> random_x(0, SIZE - 1);          // x in [0..SIZE-1]
        uniform_int_distribution<int> random_y(0, SIZE - 1);          // y in [0..SIZE-1]
        uniform_int_distribution<int> random_len(p.L_min, p.L_max);   // length in [L_min..L_max]
        uniform_int_distribution<int> random_press(p.press_min, p.press_max); // press in [press_min..press_max]
        uniform_real_distribution<double> u(0.0, 1.0);                // real in [0..1]
        uniform_int_distribution<int> coin(0, 1);                     // random bit

        // Start point is uniform random on canvas.
        this->x_start = random_x(rng);
        this->y_start = random_y(rng);

        // Sample two lengths for x and y components.
        int lenx = random_len(rng);
        int leny = random_len(rng);

        // Endpoints are computed by adding/subtracting lengths, then clamped.
        this->x_finish = coin(rng) ? max(0, this->x_start - lenx) : min(this->x_start + lenx, SIZE - 1);
        this->y_finish = coin(rng) ? max(0, this->y_start - leny) : min(this->y_start + leny, SIZE - 1);

        // Randomly choose whether this stroke is pencil or eraser.
        this->type = (u(rng) < p.erase_prob) ? 1 : 0;

        // Shade depends on type:
        // pencil can be dark/medium; eraser is near white.
        if (this->type == 0) {
            uniform_int_distribution<int> pencil_shade(0, PENCIL_SHADE_MAX);
            this->shade = pencil_shade(rng);
        } else {
            uniform_int_distribution<int> er_shade(ERASER_SHADE_MIN, 255);
            this->shade = er_shade(rng);
        }

        // Pressure (alpha strength) is random within scheduled range.
        this->press = random_press(rng);

        // Ensure validity (clamp everything).
        this->correctionLine();
    }

    // Constructor for drawing (coordinate shift).
    // Used in delta_fitness: we evaluate a line only inside its bounding box,
    // so we need line coordinates relative to that box (subtract x0, y0).
    Line(Line* line, int x0, int y0) {
        this->x_start = line->get_x_start() - x0;
        this->x_finish = line->get_x_finish() - x0;
        this->y_start = line->get_y_start() - y0;
        this->y_finish = line->get_y_finish() - y0;
        this->shade = line->get_shade();
        this->press = line->get_press();
        this->type = line->get_type();
    }

    // Clamp all line parameters into valid ranges.
    void correctionLine() {
        this->x_start = correction(this->x_start, 0, SIZE - 1);
        this->x_finish = correction(this->x_finish, 0, SIZE - 1);
        this->y_start = correction(this->y_start, 0, SIZE - 1);
        this->y_finish = correction(this->y_finish, 0, SIZE - 1);
        this->shade = correction(this->shade, 0, 255);
        this->press = correction(this->press, 0, 255);
        this->type = (this->type == 1) ? 1 : 0;  // forces type to be either 0 or 1
    }

    // Simple getters (const-correct).
    int get_x_start() const { return this->x_start; }
    int get_x_finish() const { return this->x_finish; }
    int get_y_start() const { return this->y_start; }
    int get_y_finish() const { return this->y_finish; }
    int get_shade() const { return this->shade; }
    int get_press() const { return this->press; }
    int get_type() const { return this->type; }

    // Mutation operator:
    // - adds Gaussian noise to endpoints, shade, press
    // - sometimes resets to a new random line
    // - sometimes flips pencil/eraser
    Line mutation(const Params& p) const {
        Line out = *this; // start from current chromosome


        // Gaussian mutations (0 mean).
        normal_distribution<double> dxy(0.0, (double)p.delta_xy);
        normal_distribution<double> dsh(0.0, (double)p.delta_shade);
        normal_distribution<double> dpr(0.0, (double)p.delta_press);
        uniform_real_distribution<double> u(0.0, 1.0);

        // Reset mutation: with some probability, create a new random line.
        // This is a "diversity injection" mechanism.
        if (u(rng) < p.reset_prob) return Line(p);

        // Apply noise to geometry.
        out.x_start += (int)lround(dxy(rng));
        out.x_finish += (int)lround(dxy(rng));
        out.y_start += (int)lround(dxy(rng));
        out.y_finish += (int)lround(dxy(rng));

        // Apply noise to shade and press.
        out.shade += (int)lround(dsh(rng));
        out.press += (int)lround(dpr(rng));

        // With small probability, flip type (pencil / eraser).
        if (u(rng) < p.delta_type_prob) out.type = (out.type == 0) ? 1 : 0;

        // Clamp again.
        out.correctionLine();
        return out;
    }

    // Crossover operator:
    // - uniform crossover per field (each gene comes from parent a or b with prob 0.5).
    static Line crossover(const Line& a, const Line& b) {
        uniform_int_distribution<int> coin(0, 1);

        Line c;
        c.x_start = coin(rng) ? a.x_start : b.x_start;
        c.x_finish = coin(rng) ? a.x_finish : b.x_finish;
        c.y_start = coin(rng) ? a.y_start : b.y_start;
        c.y_finish = coin(rng) ? a.y_finish : b.y_finish;
        c.shade = coin(rng) ? a.shade : b.shade;
        c.press = coin(rng) ? a.press : b.press;
        c.type = coin(rng) ? a.type : b.type;

        c.correctionLine();
        return c;
    }
};

// Drawing operator
// Key idea: we draw a binary mask of the line and blend only pixels on that mask.
static void add_line(cv::Mat& image, Line* line) {
    // Convert press (0..255) into alpha in [0..1].
    // Higher press means the stroke changes pixels more strongly.
    float alpha = float(line->get_press()) / 255.0f;

    // Create a mask of the same size as the canvas.
    // CV_8U means 8-bit unsigned; 0 = not affected, 255 = affected.
    cv::Mat mask(image.size(), CV_8U, cv::Scalar(0));

    // Define endpoints for the line.
    cv::Point p1(line->get_x_start(), line->get_y_start());
    cv::Point p2(line->get_x_finish(), line->get_y_finish());

    // Draw the line into the mask with value 255.
    // Thickness = 1 pixel, LINE_8 = non-antialiased rasterization (simpler and deterministic).
    cv::line(mask, p1, p2, cv::Scalar(255), 1, cv::LINE_8);

    // Iterate over pixels and blend only where mask is non-zero.
    for (int y = 0; y < image.rows; y++) {
        uint8_t* pix = image.ptr<uint8_t>(y);   // pointer to row y in image
        const uint8_t* m = mask.ptr<uint8_t>(y); // pointer to row y in mask

        for (int x = 0; x < image.cols; x++) {
            if (!m[x]) continue; // skip pixels not touched by the stroke

            float old_value = float(pix[x]); // current grayscale value
            float new_value;                 // updated grayscale value

            // Pencil: move pixel toward shade.
            if (line->get_type() == 0) {
                new_value = old_value + alpha * (float(line->get_shade()) - old_value);
            }
                // Eraser: move pixel toward white.
            else {
                new_value = old_value + alpha * (255.0f - old_value);
            }

            // Clamp and write back.
            pix[x] = (uint8_t)correction((int)lround(new_value), 0, 255);
        }
    }
}

// Fitness model
// Base is used only to compute fitness.
class Error {
private:
    cv::Mat base; // grayscale target image, resized to SIZE x SIZE (CV_8UC1)



public:
    // Load input image, convert to grayscale, resize, store to base.
    Error() {
        filesystem::path pathToImage = working_dir / ("input" + to_string(inputIndex) + ".jpg");
        cv::Mat picture = cv::imread(pathToImage.string(), cv::IMREAD_COLOR);
        if (picture.empty()) {
            throw std::runtime_error("Cannot read the file: " + pathToImage.string());
        }

        cv::Mat gray;
        cv::cvtColor(picture, gray, cv::COLOR_BGR2GRAY);                 // BGR to grayscale
        cv::resize(gray, gray, cv::Size(SIZE, SIZE), 0, 0, cv::INTER_AREA); // resize to working resolution
        gray.convertTo(this->base, CV_8U);                               // ensure CV_8UC1
    }

    // Full-image fitness on the whole canvas.
    double full_fitness(const cv::Mat& image) {
        return fitness(image, this->base);
    }

    // MSE-like objective (SSE / N). Lower is better.
    // Here we manually compute squared error to avoid extra OpenCV operations.
    static double fitness(const cv::Mat& image, const cv::Mat& base) {
        double sse = 0.0;
        for (int y = 0; y < image.rows; y++) {
            const uint8_t* a = image.ptr<uint8_t>(y); // canvas row
            const uint8_t* b = base.ptr<uint8_t>(y);  // target row
            for (int x = 0; x < image.cols; x++) {
                double d = double(b[x]) - double(a[x]);
                sse += d * d;
            }
        }
        return sse / double(image.total()); // average squared error (lower is better)
    }

    // Delta fitness (local improvement estimate) for a candidate stroke.
    // Computes: prev_f - new_f on a small box around the line.
    //
    // This method DOES use the target/base to evaluate fitness for candidates.
    double delta_fitness(const cv::Mat& image, Line* line) {
        // Compute bounding box around the stroke (with small padding).
        int box_x_start = max(0, min(line->get_x_start(), line->get_x_finish()) - 2);
        int box_y_start = max(0, min(line->get_y_start(), line->get_y_finish()) - 2);
        int box_x_finish = min(SIZE - 1, max(line->get_x_start(), line->get_x_finish()) + 2);
        int box_y_finish = min(SIZE - 1, max(line->get_y_start(), line->get_y_finish()) + 2);

        int box_width = box_x_finish - box_x_start + 1;
        int box_height = box_y_finish - box_y_start + 1;

        // Safety check (should usually not happen, but keeps code robust).
        if (box_width <= 0 || box_height <= 0) return 0.0;

        // Define box rectangle.
        cv::Rect box(box_x_start, box_y_start, box_width, box_height);

        // Extract box from target and canvas.
        cv::Mat box_base = base(box);            // view into base (no copy)
        cv::Mat box_image = image(box).clone();  // copy because we will modify it for evaluation

        // Compute fitness before applying stroke (box only).
        double prev_f = fitness(box_image, box_base);

        // Shift line coordinates into box coordinate system and apply stroke to copy.
        Line drawing_line = Line(line, box_x_start, box_y_start);
        add_line(box_image, &drawing_line);

        // Compute new fitness after stroke.
        double new_f = fitness(box_image, box_base);

        // Return improvement amount.
        return prev_f - new_f;
    }
};

// Canvas wrapper (current image + its full fitness)
class Canvac {
private:
    cv::Mat image;   // current canvas (CV_8UC1 grayscale)
    double fitness;  // current full fitness (lower is better)
    Error* error;    // pointer to fitness evaluator (has target)

public:
    // Initialize blank white canvas and compute initial fitness.
    Canvac(Error* error) {
        this->error = error;
        this->image = cv::Mat(SIZE, SIZE, CV_8UC1, cv::Scalar(255)); // start from white noise-free canvas
        this->fitness = error->full_fitness(this->image);
    }

    // Returns current image (copy-on-write Mat; cheap return).
    cv::Mat get_image() { return this->image; }


    // Returns current full-image fitness.
    double get_fitness() const { return this->fitness; }

    // Apply one stroke to the canvas and refresh full fitness.
    double draw_line(Line* line) {
        add_line(this->image, line);
        this->fitness = this->error->full_fitness(this->image);
        return this->fitness;
    }

    // Access to fitness evaluator (Error).
    Error* get_error() { return this->error; }

    // Save current canvas image to disk.
    void save_image(const filesystem::path& path) {
        filesystem::create_directories(path.parent_path());
        vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 };

        bool ok = cv::imwrite(path.string(), this->image, params);
        if (!ok) throw std::runtime_error("save_image: cannot save " + path.string());
    }
};

// Evolutionary Algorithm (EA) to choose the next stroke
// This EA tries to find ONE good stroke given the current canvas.
// Chromosome = Line
// Fitness = delta_fitness, higher is better.
class EA {
private:
    vector<pair<Line, double>> population; // each individual is (Line, score=delta_fitness)
    Canvac* canvac;                        // current canvas
    Params params;                         // scheduled parameters for this stage

public:
    // Initialize population randomly and pre-score each individual via delta_fitness.
    EA(Canvac* canvac, const Params& p) {
        this->canvac = canvac;
        this->params = p;

        cv::Mat image = canvac->get_image();

        population.reserve(params.population_size);

        for (int i = 0; i < params.population_size; i++) {
            Line line(params); // random chromosome
            double score = canvac->get_error()->delta_fitness(image, &line); // improvement estimate
            population.push_back(make_pair(line, score));
        }

        // Sort so population[0] is best candidate (highest improvement).
        sort(population.begin(), population.end(), [](const pair<Line, double>& a, const pair<Line, double>& b) { return a.second > b.second; });
    }

    // Run one tournament selection: pick "tournament_participants" random individuals and return best index.
    int tournament_one() {
        uniform_int_distribution<int> pick(0, population.size() - 1);

        int best = pick(rng);
        double best_score = population[best].second;

        for (int i = 0; i < params.tournament_participants - 1; i++) {
            int cur = pick(rng);
            double cur_score = population[cur].second;
            if (cur_score > best_score) {
                best_score = cur_score;
                best = cur;
            }
        }
        return best;
    }

    // Build a parent pool by running tournament "tournament_winner" times.
    vector<int> tournament() {
        vector<int> winners;
        winners.reserve(params.tournament_winner);
        for (int i = 0; i < params.tournament_winner; i++) winners.push_back(tournament_one());
        return winners;
    }

    // One EA generation step:
    // - select parent indices
    // - create children by crossover + mutation
    // - score them by delta_fitness
    // - keep best POPULATION_SIZE
    void iteration() {
        cv::Mat image = canvac->get_image();

        vector<int> selection = tournament();                 // indices into population
        uniform_int_distribution<int> pick(0, selection.size() - 1);

        vector<pair<Line, double>> next;
        next.reserve(params.population_size + params.child_number);

        // Keep the entire old population.
        sort(population.begin(), population.end(),
             [](const pair<Line, double>& a, const pair<Line, double>& b) { return a.second > b.second; });

        for (int i = 0; i < population.size(); i++) {
            next.push_back(population[i]);
        }

        // Generate children.
        for (int i = 0; i < params.child_number; i++) {
            int p1 = selection[pick(rng)];
            int p2 = selection[pick(rng)];


            Line child = Line::crossover(population[p1].first, population[p2].first);
            child = child.mutation(params);

            double score = canvac->get_error()->delta_fitness(image, &child);
            next.push_back(make_pair(child, score));
        }

        // Sort all candidates and keep only top population_size.
        sort(next.begin(), next.end(),
             [](const pair<Line, double>& a, const pair<Line, double>& b) { return a.second > b.second; });

        if (next.size() > params.population_size) next.resize(params.population_size);

        population = next; // replace population
    }

    // Run EA for a fixed number of generations, then apply the best line to the canvas.
    void run_EA() {
        for (int i = 0; i < params.generations; i++) {
            iteration();
        }

        // Best individual is at index 0.
        Line best = population[0].first;
        canvac->draw_line(&best);
    }
};

// Buffer of last N images
// This is used to save the LAST_KEEP last results of each run.
//  Because each iteration improves the picture,
//  therefore the top five are the last five.
static void push_last_images(vector<cv::Mat>& last_images, const cv::Mat& img) {
    last_images.push_back(img.clone()); // clone so buffer remains unchanged if canvas changes later
    if (last_images.size() > LAST_KEEP) {
        last_images.erase(last_images.begin()); // remove oldest
    }
}

// Save buffered images into the run directory with required naming pattern.
static void save_last_images(const vector<cv::Mat>& last_images, const filesystem::path& run_dir, const int run) {
    vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 };

    for (int i = 0; i < last_images.size(); i++) {
        filesystem::path p =
                run_dir / ("TimurKagarmanovOutput" + to_string(inputIndex) + "_" + to_string(run) + "_" + to_string(i + 1) + ".jpg");
        cv::imwrite(p.string(), last_images[i], params);
    }
}

// Among last N images, find the one with the best full fitness (lowest MSE).
static int find_best_last_index(const vector<cv::Mat>& last_images, Error& error) {
    int best_i = 0;
    double best_fitness = error.full_fitness(last_images[0]);

    for (int i = 1; i < last_images.size(); i++) {
        double fitness = error.full_fitness(last_images[i]);
        if (fitness < best_fitness) {
            best_fitness = fitness;
            best_i = i;
        }
    }
    return best_i;
}

// Main experiment driver
int main() {
    // Ensure base working directory exists.
    filesystem::create_directories(working_dir);

    // Disable OpenCV plugins and parallelism.
    disable_opencv_plugins_and_parallel();

    // Create target fitness model (loads and preprocesses input image).
    Error error;

    // Folder where the best-of-last-5 of each run is stored.
    filesystem::path best_dir = working_dir / "best";
    filesystem::create_directories(best_dir);

    // Repeat for each independent run.
    for (int j = 0; j < RUNS; j++) {

        // Per-run directory.
        filesystem::path run_dir = working_dir / ("run_" + to_string(j+1));
        filesystem::create_directories(run_dir);

        // Per-run log file (fitness curves are plotted later in Python).
        ofstream log_file(run_dir / "logs.txt", ios::out);
        log_file << "time_sec lines best_fitness avg_fitness\n";

        // Initialize blank canvas for this run.
        Canvac canvac(&error);

        // t0: start time point for runtime limit & timestamps.
        auto t0 = chrono::steady_clock::now();

        int lines = 0;                 // how many strokes were applied to this canvas
        double sum_fitness_block = 0.0; // sum of full fitness values for averaging over LOG_EVERY_LINES

        // Buffer to store last LAST_KEEP snapshots.
        vector<cv::Mat> last_images;
        last_images.reserve(LAST_KEEP);


        // Main loop: stop by stroke limit or time limit.
        while (lines < LINE_NUMBER) {
            auto now = chrono::steady_clock::now();
            double sec = chrono::duration<double>(now - t0).count();
            if (sec >= RUN_TIME_LIMIT_SEC) {
                break; // time limit reached
            }

            // Progress ratio for scheduled parameters.
            double t = (double)lines / (double)(LINE_NUMBER - 1);
            t = min(1.0, max(0.0, t));

            // Get current Params based on progress.
            Params params = get_params(t);

            // Create EA for "next stroke" using current params.
            EA ea(&canvac, params);

            // Run EA and apply one best stroke to canvas.
            ea.run_EA();

            // Increase stroke count.
            lines++;

            // Current full-image fitness after applying stroke.
            double current_fitness = canvac.get_fitness();

            // Accumulate fitness values for block average.
            sum_fitness_block += current_fitness;

            // Save snapshot to the last-N buffer.
            push_last_images(last_images, canvac.get_image());

            // Write log entry each LOG_EVERY_LINES strokes.
            if (lines % LOG_EVERY_LINES == 0) {
                auto now_log = chrono::steady_clock::now();
                double sec_log = chrono::duration<double>(now_log - t0).count();

                // Average fitness over the last LOG_EVERY_LINES strokes.
                double avr_fitness = sum_fitness_block / double(LOG_EVERY_LINES);

                // Reset block accumulator.
                sum_fitness_block = 0.0;

                log_file << sec_log << " " << lines << " " << canvac.get_fitness() << " " << avr_fitness << "\n";
                log_file.flush();
            }
        }

        // Save the last 5 snapshots for this run.
        save_last_images(last_images, run_dir, j + 1);

        // Pick the best among last 5 and save it into working_dir/best.
        if (!last_images.empty()) {
            int best_i = find_best_last_index(last_images, error);
            filesystem::path out_best =
                    best_dir / ("TimurKagarmanovOutput" + to_string(inputIndex) + "_" + to_string(j + 1) + "_" + to_string(best_i + 1) + ".jpg");

            vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 };
            cv::imwrite(out_best.string(), last_images[best_i], params);
        }

        // Final runtime statistics for end-of-run entry.
        auto t1 = chrono::steady_clock::now();
        double total_sec = chrono::duration<double>(t1 - t0).count();

        // End line for logs.
        log_file << "END " << total_sec << " " << lines << " " << canvac.get_fitness() << "\n";
        log_file.close();
    }

    return 0;
}
