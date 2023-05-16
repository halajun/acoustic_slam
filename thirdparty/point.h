#ifndef Point_H_
#define Point_H_
#include <assert.h>

const int DIM = 2;


class PointD2 {
   float coord[DIM];
   public:
   PointD2() {}
   PointD2(float x, float y) {
      coord[0] = x;
      coord[1] = y;
   }

   PointD2 operator+(PointD2 b);
   PointD2 operator-(PointD2 b);
   PointD2 operator*(float lambda);
   PointD2 operator/(float lambda);
   float operator*(PointD2 b);
   float Norm();
   PointD2 Rotate(float angle);
   float  operator[](int i) const;
   float& operator [](int i );
};



PointD2 PointD2::operator+(PointD2 b) {
   PointD2 c;
   for (int i=0;i<DIM;i++)
      c.coord[i] = coord[i] + b.coord[i];
   return c;
}

PointD2 PointD2::operator-(PointD2 b) {
   PointD2 c;
   for (int i=0;i<DIM;i++)
      c.coord[i] = coord[i] - b.coord[i];
   return c;
}

PointD2 PointD2::operator*(float lambda) {
   PointD2 c;
   for (int i=0;i<DIM;i++)
      c.coord[i] = coord[i] * lambda;
   return c;
}

PointD2 PointD2::operator/(float lambda) {
   PointD2 c;
   for (int i=0;i<DIM;i++)
      c.coord[i] = coord[i] / lambda;
   return c;
}

float PointD2::operator*(PointD2 b) {
   float sum = 0;
   for (int i=0;i<DIM;i++)
      sum += coord[i] * b.coord[i];
   return sum;
}

float PointD2::Norm() {
   float sum = 0;
   for (int i=0;i<DIM;i++)
      sum += coord[i] * coord[i];
   return sqrt(sum);
}

float PointD2::operator []( int i) const {
   assert ( i>=0 || i < DIM);
   return coord[i]; 
}

float& PointD2::operator []( int i) {
   assert ( i>=0 || i < DIM);
   return coord[i]; 
}

#endif /* Point_H_ */
