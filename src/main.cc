#include <iostream>
#include <fstream>
#include <string>
#include <cmath>      
#include <vector>      
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
        void toPPM()
        {
             std::ofstream myfile;
             myfile.open ("example.ppm");
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
        double ** m;
    public:
        Matrix(int rows, int cols)
        {
            this->rows = rows;
            this->cols = cols;
            m = new double*[rows];
            for(int i = 0; i < rows; i ++){
                m[i] = new double[cols]; 
                for(int j = 0; j < cols; j ++){
                    m[i][j] = 0.0;
                }
            }
        }
        Matrix(int rows, int cols, double * arr)
        {
            this->rows = rows;
            this->cols = cols;
            size_t n = sizeof(arr)/sizeof(arr[0]);
            assert(rows + cols != n);

            m = new double*[rows];
            for(int i = 0; i < rows; i ++){
                m[i] = new double[cols];
                for(int j = 0; j < cols; j ++){
                    m[i][j] = arr[i + j];
                }
            }
        }
        double ** getMatrix()
        {
            return m;
        }
        double* &operator[](int index){
            return m[index];
        }
        Matrix operator *(const Matrix &other){
            assert(other.rows != this.rows || other.cols != this.cols || other.rows != this.rows || other.cols != this.cols);
            Matrix result = Matrix(rows, cols);
            for(int row = 0; row < rows; row ++){
                for(int col = 0; col < cols; col ++){
                    for(int k = rows-1; k >= 0; k --){
                        result[row][col] += m[row][k] * other.m[k][col];
                    }
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

Projectile tick(Environment env, Projectile proj)
{
    Point position = to_point(proj.getPosition() + proj.getVelocity());
    Vector velocity = to_vector(proj.getVelocity() + env.getGravity() + env.getWind());
    return Projectile(position, velocity);
}

int main(int argc, char * argv[])
{
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
    c.toPPM();
    double arr[4] = {1,1,1,1};
    Matrix m1 = Matrix(2,2,arr);
    Matrix m2 = Matrix(2,2,arr);
    Matrix m3 = m1 * m2;
    std::cout << m3.toString() << std::endl;
    return 0;
}