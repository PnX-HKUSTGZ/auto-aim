#include "rune_detector/rune_detector.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <opencv2/highgui.hpp>

#include "rune_detector/types.hpp"

namespace rm_auto_aim
{

RuneDetector::RuneDetector(const Params & params) : params_(params) {}

std::vector<RuneObject> RuneDetector::detectRune(const cv::Mat & img, int min_lightness)
{
    const cv::Mat bin = makeBinary(img, min_lightness);
    const auto contours = findValidContours(bin);

    auto centers = classifyCenters(contours);
    auto targets_inactive = classifyTargets(contours, false);
    auto targets_active = classifyTargets(contours, true);
    auto fans_inactive = classifyFans(contours, false);
    auto fans_active = classifyFans(contours, true);

    const auto center_pt = chooseCenter(centers, targets_inactive, fans_inactive);
    if (center_pt.x < 0.f || center_pt.y < 0.f)
        return {};

    const auto combos = buildCombos(targets_inactive, targets_active, fans_inactive, fans_active, center_pt);

    // 只输出含未激活靶心的组合
    std::vector<RuneObject> output;
    for (const auto & combo : combos)
    {
        const auto & target = std::get<0>(combo);
        if (target.area <= 0.0)
            continue;
        RuneObject obj;
        obj.type = RuneType::INACTIVATED;
        Candidate center_candidate{cv::RotatedRect(center_pt, cv::Size2f(1.f, 1.f), 0.f), 1.0};
        const auto pts = toFeaturePoints(center_candidate, target, std::get<2>(combo));
        obj.pts = pts;
        output.push_back(obj);
    }
    return output;
}

cv::Mat RuneDetector::makeBinary(const cv::Mat & rgb, int min_lightness) const
{
    cv::Mat bin;
    std::vector<cv::Mat> ch;
    cv::split(rgb, ch);
    cv::Mat diff;
    if (detect_color == EnemyColor::RED)
    {
        cv::subtract(ch[2], ch[0], diff);
        cv::threshold(diff, bin, params_.gray_threshold_red, 255, cv::THRESH_BINARY);
    }
    else
    {
        cv::subtract(ch[0], ch[2], diff);
        cv::threshold(diff, bin, params_.gray_threshold_blue, 255, cv::THRESH_BINARY);
    }
    // 亮度兜底
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    cv::Mat bin_light;
    cv::threshold(gray, bin_light, min_lightness, 255, cv::THRESH_BINARY);
    cv::bitwise_or(bin, bin_light, bin);
    return bin;
}

std::vector<std::vector<cv::Point>> RuneDetector::findValidContours(const cv::Mat & bin) const
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> filtered;
    for (auto & c : contours)
    {
        const double a = cv::contourArea(c);
        if (a < params_.min_contour_area || a > params_.max_contour_area)
            continue;
        filtered.push_back(std::move(c));
    }
    return filtered;
}

std::vector<RuneDetector::Candidate> RuneDetector::classifyCenters(const std::vector<std::vector<cv::Point>> & contours) const
{
    std::vector<Candidate> res;
    for (const auto & c : contours)
    {
        const double area = cv::contourArea(c);
        if (area < params_.center_min_area || area > params_.center_max_area)
            continue;
        if (area < params_.min_contour_area)
            continue;
        const auto rect = cv::minAreaRect(c);
        const double ar = aspect(rect);
        if (ar < params_.center_min_side_ratio || ar > params_.center_max_side_ratio)
            continue;
        const double rness = roundness(c);
        if (rness < params_.center_min_roundness || rness > params_.center_max_roundness)
            continue;
        res.push_back({rect, area});
    }
    return res;
}

std::vector<RuneDetector::Candidate> RuneDetector::classifyTargets(const std::vector<std::vector<cv::Point>> & contours, bool active) const
{
    std::vector<Candidate> res;
    for (const auto & c : contours)
    {
        const double area = cv::contourArea(c);
        const double min_a = active ? params_.target_active_min_area : params_.target_inactive_min_area;
        const double max_a = active ? params_.target_active_max_area : params_.target_inactive_max_area;
        if (area < min_a || area > max_a)
            continue;
        const auto rect = cv::minAreaRect(c);
        const double ar = aspect(rect);
        const double min_ar = active ? params_.target_active_min_side_ratio : params_.target_inactive_min_side_ratio;
        const double max_ar = active ? params_.target_active_max_side_ratio : params_.target_inactive_max_side_ratio;
        if (ar < min_ar || ar > max_ar)
            continue;
        const double area_ratio = area / rect.size.area();
        const double min_area_ratio = active ? params_.target_active_min_area_ratio : params_.target_inactive_min_area_ratio;
        const double max_area_ratio = active ? params_.target_active_max_area_ratio : params_.target_inactive_max_area_ratio;
        if (area_ratio < min_area_ratio || area_ratio > max_area_ratio)
            continue;
        res.push_back({rect, area});
    }
    return res;
}

std::vector<RuneDetector::Candidate> RuneDetector::classifyFans(const std::vector<std::vector<cv::Point>> & contours, bool active) const
{
    std::vector<Candidate> res;
    for (const auto & c : contours)
    {
        const double area = cv::contourArea(c);
        const double min_a = active ? params_.fan_active_min_area : params_.fan_inactive_min_area;
        const double max_a = active ? params_.fan_active_max_area : params_.fan_inactive_max_area;
        if (area < min_a || area > max_a)
            continue;
        const auto rect = cv::minAreaRect(c);
        const double ar = aspect(rect);
        const double max_ar = active ? params_.fan_active_max_side_ratio : params_.fan_inactive_max_side_ratio;
        const double min_ar = active ? 1.0 : params_.fan_inactive_min_side_ratio;
        if (ar < min_ar || ar > max_ar)
            continue;
        const double area_ratio = area / rect.size.area();
        if (active)
        {
            if (area_ratio < params_.fan_active_min_area_ratio || area_ratio > params_.fan_active_max_area_ratio)
                continue;
            const double peri = cv::arcLength(c, true);
            const double ap = area / (peri * peri + 1e-6);
            if (ap < params_.fan_active_min_area_peri_ratio || ap > params_.fan_active_max_area_peri_ratio)
                continue;
        }
        res.push_back({rect, area});
    }
    return res;
}

cv::Point2f RuneDetector::chooseCenter(const std::vector<Candidate> & centers, const std::vector<Candidate> & targets, const std::vector<Candidate> & fans) const
{
    if (!centers.empty())
    {
        return std::max_element(centers.begin(), centers.end(), [](const Candidate & a, const Candidate & b) { return a.area < b.area; })->rect.center;
    }
    // fallback: average of targets/fans
    std::vector<cv::Point2f> pts;
    for (auto & t : targets) pts.push_back(t.rect.center);
    for (auto & f : fans) pts.push_back(f.rect.center);
    if (pts.empty()) return {-1.f, -1.f};
    cv::Point2f sum(0.f,0.f);
    for (auto & p : pts) sum += p;
    return sum * (1.f / static_cast<float>(pts.size()));
}

std::vector<std::tuple<RuneDetector::Candidate, RuneDetector::Candidate, RuneDetector::Candidate>> RuneDetector::buildCombos(const std::vector<Candidate> & targets_inactive, const std::vector<Candidate> & targets_active, const std::vector<Candidate> & fans_inactive, const std::vector<Candidate> & fans_active, const cv::Point2f & center) const
{
    std::vector<std::tuple<Candidate, Candidate, Candidate>> combos;

    auto nearestFan = [&](const Candidate & target, const std::vector<Candidate> & fans) -> Candidate
    {
        if (fans.empty()) return {};
        const double max_dist = params_.max_distance_ratio * rectLongSide(target.rect);
        double best = 1e9;
        Candidate sel{};
        for (const auto & f : fans)
        {
            double d = cv::norm(f.rect.center - target.rect.center);
            if (d < best && d < max_dist)
            {
                best = d;
                sel = f;
            }
        }
        return sel;
    };

    for (const auto & t : targets_inactive)
    {
        auto f = nearestFan(t, fans_inactive);
        combos.emplace_back(t, Candidate{cv::RotatedRect(center, cv::Size2f(1.f,1.f), 0.f), 1.0}, f);
    }
    // 按角度排序稳定索引
    std::sort(combos.begin(), combos.end(), [&](const auto & a, const auto & b) {
        auto ang = [&](const Candidate & c) {
            cv::Point2f dir = c.rect.center - center;
            return std::atan2(dir.y, dir.x);
        };
        return ang(std::get<0>(a)) < ang(std::get<0>(b));
    });
    return combos;
}

double RuneDetector::rectLongSide(const cv::RotatedRect & r)
{
    return std::max(r.size.width, r.size.height);
}

double RuneDetector::aspect(const cv::RotatedRect & r)
{
    const double a = std::max(r.size.width, r.size.height);
    const double b = std::min(r.size.width, r.size.height);
    if (b < 1e-3) return 1.0;
    return a / b;
}

double RuneDetector::roundness(const std::vector<cv::Point> & c)
{
    const double area = cv::contourArea(c);
    const double peri = cv::arcLength(c, true);
    if (peri < 1e-3) return 0.0;
    return 4.0 * CV_PI * area / (peri * peri);
}

cv::Point2f RuneDetector::rectDir(const cv::RotatedRect & r)
{
    const float angle = static_cast<float>(r.angle * CV_PI / 180.0);
    return {std::cos(angle), std::sin(angle)};
}

void RuneDetector::rectLongEndpoints(const cv::RotatedRect & r, cv::Point2f & a, cv::Point2f & b)
{
    cv::Point2f pts[4];
    r.points(pts);
    double maxd = -1;
    a = pts[0];
    b = pts[1];
    for (int i = 0; i < 4; ++i)
    {
        for (int j = i + 1; j < 4; ++j)
        {
            double d = cv::norm(pts[i] - pts[j]);
            if (d > maxd)
            {
                maxd = d;
                a = pts[i];
                b = pts[j];
            }
        }
    }
}

FeaturePoints RuneDetector::toFeaturePoints(const Candidate & center, const Candidate & target, const Candidate & fan)
{
    FeaturePoints pts;
    pts.r_center = center.rect.center;

    if (fan.area > 0.0)
    {
        cv::Point2f a, b;
        rectLongEndpoints(fan.rect, a, b);
        // arm: bottom = farther from center
        if (cv::norm(a - center.rect.center) > cv::norm(b - center.rect.center))
        {
            pts.arm_bottom = a;
            pts.arm_top = b;
        }
        else
        {
            pts.arm_bottom = b;
            pts.arm_top = a;
        }
    }

    if (target.area > 0.0)
    {
        cv::Point2f corners[4];
        target.rect.points(corners);
        // sort by x then y to map to left/right/top/bottom roughly
        std::array<cv::Point2f,4> arr{corners[0],corners[1],corners[2],corners[3]};
        std::sort(arr.begin(), arr.end(), [](const cv::Point2f & p1, const cv::Point2f & p2){ return p1.x < p2.x;});
        auto left1 = arr[0];
        auto left2 = arr[1];
        auto right1 = arr[2];
        auto right2 = arr[3];
        pts.hit_left = (left1.y < left2.y) ? left1 : left2;
        pts.hit_bottom = (left1.y > left2.y) ? left1 : left2;
        pts.hit_right = (right1.y < right2.y) ? right1 : right2;
        pts.hit_top = (right1.y > right2.y) ? right1 : right2;
    }
    return pts;
}

std::tuple<cv::Point2f, cv::Mat> RuneDetector::detectRTag(const cv::Mat & img, const cv::Point2f & prior)
{
    if (prior.x < 0 || prior.x > img.cols || prior.y < 0 || prior.y > img.rows)
    {
        return {prior, cv::Mat::zeros(cv::Size(200, 200), CV_8UC3)};
    }

    cv::Rect roi = cv::Rect(prior.x - 100, prior.y - 100, 200, 200) & cv::Rect(0, 0, img.cols, img.rows);
    const cv::Point2f prior_in_roi = prior - cv::Point2f(roi.tl());

    cv::Mat img_roi = img(roi);
    cv::Mat gray_img;
    cv::cvtColor(img_roi, gray_img, cv::COLOR_BGR2GRAY);
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binary_img, binary_img, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    auto it = std::find_if(contours.begin(), contours.end(), [p = prior_in_roi](const std::vector<cv::Point> & contour) -> bool {
        return cv::boundingRect(contour).contains(p);
    });

    cv::cvtColor(binary_img, binary_img, cv::COLOR_GRAY2BGR);

    if (it == contours.end())
    {
        return {prior, binary_img};
    }

    cv::drawContours(binary_img, contours, static_cast<int>(it - contours.begin()), cv::Scalar(0, 255, 0), 2);

    cv::Point2f center = std::accumulate(
        it->begin(),
        it->end(),
        cv::Point2f(0.f, 0.f),
        [](const cv::Point2f & acc, const cv::Point & p) {
            return acc + static_cast<cv::Point2f>(p);
        });
    center *= 1.f / static_cast<float>(it->size());
    center += cv::Point2f(roi.tl());

    return {center, binary_img};
}

} // namespace rm_auto_aim