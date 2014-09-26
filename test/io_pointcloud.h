// This file is part of BilateralBlueNoisePointcloudSampling (BBNPS).
//
// Copyright (C) 2014 Christoph Heindl <christoph.heindl@gmail.com>
//
// BBNPS is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// BBNPS is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include <Eigen/Dense>
#include <stdio.h>
#include <vector>

/** Load oriented point cloud from file in XYZ format. 
    Each row in the file is composed of the following six values px py pz nx ny nz and describes a single point/normal pair. */
bool loadPointcloudFromXYZFile(
                            const char *path,
							std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &points,
							std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &normals)
{
    static const char *format = "%f %f %f %f %f %f";
    
    points.clear();
    normals.clear();
 
    FILE *f = fopen(path, "r");
    if (f == 0) {
        return false;
    }
    
    const int buf_size = 256;
    char buffer[buf_size];
    
    Eigen::Vector3f p, n;
    while (fscanf(f, format, &p.x(), &p.y(), &p.z(), &n.x(), &n.y(), &n.z()) == 6) {
        fgets(buffer, buf_size, f); // read line remainings
        points.push_back(p);
        normals.push_back(n.normalized()); // ensure normal unit length.
    }
    
    fclose(f);
    
    return !points.empty();
}

/** Save oriented point cloud in XYZ format.
    Each row in the file is composed of the following six values px py pz nx ny nz and describes a single point/normal pair. */
bool savePointcloudToXYZFile(
                               const char *path,
							   const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &points,
							   const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &normals)
{
    static const char *format = "%g %g %g %g %g %g\n";
    
    FILE *f = fopen(path, "w");
    if (f == 0) {
        return false;
    }
    
    for(size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3f &p = points[i];
        const Eigen::Vector3f &n = normals[i];
        fprintf(f, format, p.x(), p.y(), p.z(), n.x(), n.y(), n.z());
    }
    
    fclose(f);
    
    return !points.empty();
}