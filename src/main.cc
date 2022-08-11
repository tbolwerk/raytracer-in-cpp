#include <iostream>
#include <fstream>
#include <string>
#include <cmath>      
#include <vector>      
#include <queue>
#include <assert.h>      
const double EPSILON = 0.00001;
bool equal(double a, double b)
{
    return abs(a - b) < EPSILON;
}
class Tuple {
    protected:
        double x;
        double y;
        double z;
        double w;
    public:
        Tuple(){}
        Tuple(double x, double y, double z, double w){
            this->x = x;
            this->y = y;
            this->z = z;
            this->w = w;
        }
        double getX()
        {
            return x;
        }
        double getY()
        {
            return y;
        }
        double getZ()
        {
            return z;
        }
        double getW()
        {
            return w;
        }
        double &operator[](int index){
            assert(index >= 0 && <= 4);
            switch(index){
                case 0:
                    return x;
                case 1:
                    return y;
                case 2:
                    return z;
                case 3:
                    return w;
            }
            return x;
        }
        double magnitude()
        {
            return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(w, 2));
        }
        double dot(const Tuple &other)
        {
            return (x * other.x + y * other.y + z * other.z + w * other.w);
        }
        Tuple normalize()
        {
            double magnitude = this->magnitude();
            return Tuple(x / magnitude, y / magnitude, z / magnitude, w / magnitude);
        }
        Tuple operator *(double n){
            return Tuple(x * n, y * n, z * n, w * n);
        }
        Tuple operator /(const Tuple &other)
        {
            return Tuple(x / other.x, y / other.y, z / other.z, w / other.w);
        }
        Tuple operator *(const Tuple &other)
        {
            return Tuple(x * other.x, y * other.y, z * other.z, w * other.w);
        }
        Tuple operator -()
        {
            return Tuple(-x, -y, -z, -w);
        }
        Tuple operator -(const Tuple &other)
        {
            return Tuple(x - other.x, y - other.y, z - other.z, w - other.w);
        }
        Tuple operator +(const Tuple &other)
        {
            return Tuple(x + other.x, y + other.y, z + other.z, w + other.w);
        }
        bool operator ==(const Tuple &other)
        {
            return (x == other.x && y == other.y && z == other.z && w == other.w);
        }
        std::string toString()
        {
            return "{" +  std::to_string(x) + ", " +  std::to_string(y) + ", " +  std::to_string(z) + ", " +  std::to_string(w) + "}";
        }
};

class Vector: public Tuple {
    public:
        Vector(){}
        Vector(double x, double y, double z){
            this-> x = x;
            this-> y = y;
            this-> z = z;
            this-> w = 0.0;
        }
        Vector cross(const Vector &other)
        {
            return Vector(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
        }
        Vector normalize()
        {
            double magnitude = this->magnitude();
            return Vector(x / magnitude, y / magnitude, z / magnitude);
        }
};

class Point: public Tuple {
    public:
         Point(){}
         Point(double x, double y, double z)
         {
            this->x = x;
            this->y = y;
            this->z = z;
            this->w = 1.0;
         }
};

int clamp(double a, int min, int max)
{
    int x = (int) round(a * max);
    if(x < min){
        return min;
    }
    if(x > max){
        return max;
    }
    return x;
}

class Color: public Tuple {
    private:
        double red;
        double green;
        double blue;

    public:
        Color(double red, double green, double blue)
        {
            this->x = this->red = red;
            this->y = this->green = green;
            this->z = this->blue = blue;
            this->w = 1.0;
        }
        double getRed()
        {
            return red;
        }
        double getGreen()
        {
            return green;
        }
        double getBlue()
        {
            return blue;
        }
        Color hadamardProduct(const Color &other)
        {
            return Color(red * other.red,green * other.green,blue * other.blue);
        }
        std::string toString(){
            return std::to_string(clamp(red, 0, 255)) +" " + std::to_string(clamp(green,0,255)) + " " + std::to_string(clamp(blue,0,255));
        }
};

class Canvas
{
    private:
        int width;
        int height;
        std::vector< std::vector<Color> > canvas;
    public:
        Canvas(int width, int height)
        {
            this->width = width;
            this->height = height;
            canvas = std::vector< std::vector<Color> >();
            for(int i = 0; i < width; i ++)
            {
                canvas.push_back( std::vector<Color>() );
                for(int j = 0; j < height; j ++)
                {
                    canvas[i].push_back(Color(0,0,0));
                }
            }
        }
        int getWidth()
        {
            return width;
        }
        int getHeight()
        {
            return height;
        }
        void writePixel(int x, int y, Color color)
        {
            canvas[x][y] = color;
        }
        Color pixelAt(int x, int y)
        {
            return canvas[x][y];
        }
        void toPPM(const std::string file_name)
        {
             std::ofstream myfile;
             myfile.open (file_name);
             if(myfile.is_open()){
                myfile << "P3" << std::endl;
                myfile << std::to_string(width) + " " + std::to_string(height) << std::endl;
                myfile << "255" << std::endl;
                for(int i = 0; i < height; i ++){
                    for(int j = 0; j < width; j ++){
                        myfile << pixelAt(j,i).toString() << " ";
                    }
                    myfile << std::endl;
                }
                myfile.close();
             }
        }
};

class Matrix
{
    private:
        int rows;
        int cols;
        int size;
        double ** m;
    public:
        Matrix(int rows, int cols)
        {
            this->rows = rows;
            this->cols = cols;
            this->size = (rows + cols) / 2;
            m = new double*[rows];
            empty();
        }
        Matrix(int rows, int cols, double * arr)
        {
            this->rows = rows;
            this->cols = cols;
            this->size = (rows + cols) / 2;
            size_t n = sizeof(arr)/sizeof(arr[0]);
            assert(rows + cols != n && "elements is not eq to the size");

            m = new double*[rows];
            for(int i = 0; i < rows; i ++){
                m[i] = new double[cols];
                for(int j = 0; j < cols; j ++){
                    m[i][j] = arr[i * rows + j];
                }
            }
        }
        void empty(){
             for(int i = 0; i < rows; i ++){
                m[i] = new double[cols]; 
                for(int j = 0; j < cols; j ++){
                        m[i][j] = 0.0;
                }
            }
        }
        void id(){
             for(int i = 0; i < rows; i ++){
                m[i] = new double[cols]; 
                for(int j = 0; j < cols; j ++){
                    if(i == j){
                        m[i][j] = 1.0;
                    }else{
                        m[i][j] = 0.0;
                    }
                }
            }
        }
        Matrix identity(){
            Matrix result = Matrix(4,4);
            result.id();
            return result;
        }
        Matrix copy(){
            double arr[rows * cols];
            for(int i = 0; i < rows; i ++){
                for(int j = 0; j < cols; j ++){
                    arr[i * 4 + j] = m[i][j];
                }
            }
            return Matrix(rows, cols, arr);
        }
        Matrix rotate_x(double r){
            Matrix result = this->copy();
            return Matrix::rotation_x(r) * result;
        }
        Matrix rotate_y(double r){
            Matrix result = this->copy();
            return Matrix::rotation_y(r) * result;
        }
        Matrix rotate_z(double r){
            Matrix result = this->copy();
            return Matrix::rotation_z(r) * result;
        }
        Matrix scale(double x,double y, double z){
            Matrix result = this->copy();
            return Matrix::scaling(x,y,z) * result;
        }
        Matrix translate(double x, double y, double z){
            Matrix result = this->copy();
            return Matrix::translation(x,y,z) * result;
        }
        static Matrix shearing(double x_y, double x_z, double y_x, double y_z, double z_x, double z_y){
            double arr[16] = {1,x_y,x_z,0,
                              y_x,1,y_z,0,
                              z_x,z_y,1,0,
                              0,0,0,1};
            return Matrix(4,4, arr);
        }
        static Matrix rotation_x(double r){
            double arr[16] = {1,0,0,0,
                              0,cos(r),-sin(r),0,
                              0,sin(r),cos(r),0,
                              0,0,0,1};
            return Matrix(4,4, arr);
        }
        static Matrix rotation_z(double r){
            double arr[16] = {cos(r),-sin(r),0,0,
                              sin(r),cos(r),0,0,
                              0,0,1,0,
                              0,0,0,1};
            return Matrix(4,4, arr);
        }
        static Matrix rotation_y(double r){
            double arr[16] = {cos(r),0,sin(r),0,
                              0,1,0,0,
                              -sin(r),0,cos(r),0,
                              0,0,0,1};
            return Matrix(4,4, arr);
        }
        static Matrix translation(double x, double y, double z){
            double arr[16] = {1,0,0,x,
                              0,1,0,y,
                              0,0,1,z,
                              0,0,0,1};
            return Matrix(4,4, arr);
        }
        static Matrix scaling(double x, double y, double z){
            double arr[16] = {x,0,0,0,
                              0,y,0,0,
                              0,0,z,0,
                              0,0,0,1};
            return Matrix(4,4, arr);
        }
        Matrix inverse(){
            assert(isInvertable() && "matrix is not invertable");
            Matrix m2 = Matrix(rows, cols);
            double det = determinant();
            for(int i = 0; i < rows; i ++){
                for(int j = 0; j < cols; j ++){
                    double c = cofactor(i,j);
                    m2[i][j] = c / det;
                }
            }
            return m2;
        }
        Matrix transpose(){
            Matrix result = Matrix(cols, rows);
            for(int i = 0; i < rows; i ++){
                for(int j = 0; j < cols; j ++){
                    result[i][j] = m[j][i];
                }
            }
            return result;
        }
        Matrix submatrix(int row, int col){
            Matrix result = Matrix(rows - 1, cols - 1);
            for(int i = 0; i < rows - 1; i ++){
                int x = i;
                if(i >= row){
                     x += 1;
                }
                for(int j = 0; j < cols - 1; j ++){
                    int y = j;
                    if(j >= col){
                        y += 1;
                    }
                    result[i][j] = m[x][y];
                }
            }
            return result;
        }
        double minor(int row, int col){
            return submatrix(row,col).determinant();
        }
        double cofactor(int row, int col){
            if((row + col) % 2 == 0){
                return minor(row, col);
            }
            return -minor(row,col);
        }
        double determinant(){
            double det = 0.0;
            assert(cols == rows);
            if(size == 2){
                det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            }else{
                for(int col = 0; col < size; col ++){
                    det += m[0][col] * cofactor(0, col);
                }
            }
            return det;
        }
        bool isInvertable(){
            return determinant() != 0.0;
        }
        double ** getMatrix()
        {
            return m;
        }
        double* &operator[](int index){
            return m[index];
        }
        Matrix operator *(Matrix &other){
            assert((other.rows != this.rows || other.cols != this.cols || other.rows != this.rows || other.cols != this.cols) && "rows and colums do not match");
            Matrix result = Matrix(rows, cols);
            for(int row = 0; row < rows; row ++){
                for(int col = 0; col < cols; col ++){
                    for(int k = 0; k < size; k ++){
                        result[row][col] += m[row][k] * other[k][col];
                    }
                }
            }
            return result;
        }
        Tuple operator *(Tuple &other){
            assert(rows == 4 && "rows has to be eq to 4");
            Tuple result = Tuple(0,0,0,0);
            for(int row = 0; row < rows; row ++){
                for(int col = 0; col < cols; col++){
                    result[row] += m[row][col] * other[col];
                }
            }
            return result;
        }
        std::string toString(){
            std::string result = "";
            for(int i = 0; i < rows; i ++){
                for(int j = 0; j < cols; j ++){
                    result = result + " | " + std::to_string(m[i][j]) + " | ";
                }
                result = result + "\n";
            }
            return result;
        }
};

Point to_point(Tuple tuple)
{
    return Point(tuple.getX(), tuple.getY(), tuple.getZ());
}

Vector to_vector(Tuple tuple)
{
    return Vector(tuple.getX(), tuple.getY(), tuple.getZ());
}

Color to_color(Tuple tuple)
{
    return Color(tuple.getX(), tuple.getY(), tuple.getZ());
}

class Projectile
{
    private:
        Point position;
        Vector velocity;
    public:
        Projectile(Point position, Vector velocity)
        {
            this->position = position;
            this->velocity = velocity;
        }
        Point getPosition()
        {
            return this->position;
        }
        Vector getVelocity()
        {
            return this->velocity;
        }
};

class Environment
{
    private:
        Vector gravity;
        Vector wind;
    public:
        Environment(Vector gravity, Vector wind)
        {
            this->gravity = gravity;
            this->wind = wind;
        }
        Vector getWind()
        {
            return this->wind;
        }
        Vector getGravity()
        {
            return this->gravity;
        }
};

class Ray
{
    private:
        Point origin;
        Vector direction;
    public:
        Ray(Point origin, Vector direction)
        {
            this->origin = origin;
            this->direction = direction;
        }
        Vector getDirection()
        {
            return this->direction;
        }
        Point getOrigin()
        {
            return this->origin;
        }
        Point position(double time)
        {
            return to_point(this->origin + this->direction * time);
        }
};

class Object
{
    protected:
        std::string id;
        Point position;
    public:
        std::string toString()
        {
            return "Object {" + id + " ," + "Position " + position.toString() + "}";
        }
    
};

class Intersection
{
    private:
        double time;
        Object* object;
    public:
        Intersection(double time, Object* object)
        {
            this->time = time;
            this->object = object;
        }
        double getTime(){return time;};
        Object* getObject() {return object;};
        std::string toString() const
        {
            return "{" + std::to_string(time) + ", " + object->toString() + "}";
        }
        bool operator >(const Intersection &intersection) const
        {
            return this->time > intersection.time;
        }
};

class Intersections
{
    private:
        std::priority_queue<Intersection, std::vector<Intersection>, std::greater<Intersection> > intersections;
    public:
        Intersections(std::vector<Intersection> intersections)
        {
            this->intersections = std::priority_queue<Intersection, std::vector<Intersection>, std::greater<Intersection> >(intersections.begin(), intersections.end());
        }
        std::priority_queue<Intersection, std::vector<Intersection>, std::greater<Intersection> > getIntersections(){
            return this->intersections;
        }
        void pop()
        {
            this->intersections.pop();
        }
        Intersection top()
        {
            return this->intersections.top();
        }
        bool empty()
        {
            return this->intersections.empty();
        }
};

class Hitable: public Object
{
    public:
        virtual std::vector<Intersection> insersect(Ray ray) = 0; 
};

class Sphere: public Hitable
{
    private:
        double radius;
    public:
        Sphere()
        {
            this->id = "unit_sphere";
            this->position = Point(0,0,0);
            this->radius = 1.0;
        }
        Sphere(std::string id, Point position, double radius)
        {
            this->id = id;
            this->position = position;
            this->radius = radius;
        }
        public:
            std::vector<Intersection> insersect(Ray ray)
            {
                Tuple object_to_ray = ray.getOrigin() - this->position;
                double a = ray.getDirection().dot(ray.getDirection());
                double b = 2 * ray.getDirection().dot(object_to_ray);
                double c = object_to_ray.dot(object_to_ray) - this->radius; //TODO: this->radius might be replaced with just 1

                double discriminant = pow(b,2) - 4 * a * c;

                if(discriminant < 0){
                    return std::vector<Intersection>();
                }
                double t1 = (-b - sqrt(discriminant)) / (2 * a);
                double t2 = (-b + sqrt(discriminant)) / (2 * a);
                std::vector<Intersection> v = std::vector<Intersection>();
                
                v.push_back(Intersection(t1, this));
                v.push_back(Intersection(t2, this));
                return v;
            }
};

Projectile tick(Environment env, Projectile proj)
{
    Point position = to_point(proj.getPosition() + proj.getVelocity());
    Vector velocity = to_vector(proj.getVelocity() + env.getGravity() + env.getWind());
    return Projectile(position, velocity);
}

void chapter1(){
    Point start = Point(0,1,0);
    Vector velocity = to_vector(Vector(1,1.8,0).normalize() * Vector(11.25, 11.25, 11.25));
    Projectile p = Projectile(start, velocity);

    Vector gravity = Vector(0, -0.1, 0);
    Vector wind = Vector(-0.01, 0, 0);
    Environment e = Environment(gravity, wind);

    Canvas c = Canvas(900, 550);

    while(p.getPosition().getY() > 0)
    {
        c.writePixel(p.getPosition().getX(), c.getHeight() - p.getPosition().getY(), Color(1,1,1));
        p = tick(e, p);
    }
    c.toPPM("projectile.ppm");
}

void chapter4(){
    Matrix t = Matrix(4,4).identity().translate(0,0,1);
    Tuple origin = Tuple(0,0,0,1);
    Canvas c = Canvas(400, 400);
    Tuple twelve = t * origin;
    for(int i = 0; i < 12; i ++){
        Matrix r = Matrix(4,4).identity().rotate_y(i * (M_PI / 6)).scale(150,150,150).translate(200,200,200);
        
        Tuple p = r * twelve;
        c.writePixel(p.getX(),  p.getZ(), Color(1,1,1));
    }
    c.toPPM("clock.ppm");
}

int main(int argc, char * argv[])
{
    chapter1();
    chapter4();
    Ray r = Ray(Point(0,0,-5), Vector(0,0,1));
    Sphere s = Sphere();
    std::vector<Intersection> is = s.insersect(r);
    Intersections xs = Intersections(is);

    while (!xs.empty())
    {
        std::cout << xs.top().toString() << std::endl;
        xs.pop();
    }

    return 0;
}