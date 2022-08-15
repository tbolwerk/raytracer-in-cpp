#include <cassert> 
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>      
#include <vector>      
#include <queue>
#include <optional>
const double EPSILON = 0.00001;
bool equal(double a, double b)
{
    return abs(a - b) < EPSILON;
}
double radians(double degree)
{
    return (degree / 180) * M_PI;
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
            assert(index >= 0 && index <= 4);
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
        bool operator !=(const Tuple &other)
        {
            return (x != other.x || y != other.y || z != other.z || w != other.w);
        }
        bool operator <(const Tuple &other) const
        {
            return this->x < other.x && this->y < other.y && this->z < other.z && this->w < other.w;
        }
        bool operator <=(const Tuple &other) const
        {
            return this->x <= other.x && this->y <= other.y && this->z <= other.z && this->w <= other.w;
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
        bool operator <(const Vector &other) const
        {
            return this->x < other.x && this->y < other.y && this->z < other.z;
        }
        bool operator <=(const Vector &other) const
        {
            return this->x <= other.x && this->y <= other.y && this->z <= other.z;
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
        Vector copy(){
            return Vector(x,y,z);
        }
        Vector reflect(Vector &other)
        {
            Tuple result = copy() - other * 2 * this->dot(other);
            return Vector(result.getX(), result.getY(),result.getZ());
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
        bool operator <(const Point &other) const
        {
            return this->x < other.x && this->y < other.y && this->z < other.z;
        }
        bool operator <=(const Point &other) const
        {
            return this->x <= other.x && this->y <= other.y && this->z <= other.z;
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
        Color(){}
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
        static Color white()
        {
            return Color(1,1,1);
        }
        static Color black()
        {
            return Color(0,0,0);
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
            assert(x >= 0 && x < width && "x not within bounds");
            assert(y >= 0 && y < height && "y not within bounds");
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
            return result * Matrix::rotation_x(r);
        }
        Matrix rotate_y(double r){
            Matrix result = this->copy();
            return result * Matrix::rotation_y(r);
        }
        Matrix rotate_z(double r){
            Matrix result = this->copy();
            return result * Matrix::rotation_z(r);
        }
        Matrix scale(double x,double y, double z){
            Matrix result = this->copy();
            return result * Matrix::scaling(x,y,z);
        }
        Matrix translate(double x, double y, double z){
            Matrix result = this->copy();
            return result * Matrix::translation(x,y,z);
        }
        Matrix shear(double x_y, double x_z, double y_x, double y_z, double z_x, double z_y)
        {
            Matrix result = this->copy();
            return result * Matrix::shearing(x_y, x_z, y_x, y_z, z_x, z_y);
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
                    m2[j][i] = c / det;
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
        Matrix operator *(Matrix other){
            assert((other.rows == this->rows || other.cols == this->cols) && "rows and colums do not match");
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
        Tuple operator *(Tuple other){
            assert(this->rows == 4 && "rows has to be eq to 4");
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
        Ray transform(Matrix transform)
        {
            return Ray(to_point(transform * this->origin), to_vector(transform * this->direction));
        }
};

class Material
{
    private:
        Color color;
        double ambient;
        double diffuse;
        double specular;
        double shininess;
    public:
        Material()
        {
            this->color = Color(1,1,1);
            this->ambient = 0.1;
            this->diffuse = 0.9;
            this->specular = 0.9;
            this->shininess = 200.0;
        }
        void setAmbient(double ambient)
        {
            this->ambient = ambient;
        }
        Color getColor()
        {
            return this->color;
        }
        double getAmbient()
        {
            return this->ambient;
        }
        double getDiffuse()
        {
            return this->diffuse;
        }
        double getShininess()
        {
            return this->shininess;
        }
        double getSpecular()
        {
            return this->specular;
        }
        void setColor(Color color)
        {
            this->color = color;
        }
        void setDiffuse(double diffuse)
        {
            this->diffuse = diffuse;
        }
        void setShininess(double shininess)
        {
            this->shininess = shininess;
        }
        void setSpecular(double specular)
        {
            this->specular = specular;
        }
};

class Object
{
    protected:
        std::string id;
        Point position;
        Matrix transformation = Matrix(4,4).identity();
        Material material;
    public:
        virtual Vector normalAt(Point p) = 0;
        
        std::string toString()
        {
            return "Object {" + id + " ," + "Position " + position.toString() + "}";
        }
        void setTransformation(Matrix transformation)
        {
            this->transformation = transformation;
        }
        Matrix getTransformation()
        {
            return this->transformation;
        }
        void setMaterial(Material material)
        {
            this->material = material;
        }
        Material getMaterial()
        {
            return this->material;
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
        double getTime() const {return time;};
        Object* getObject() const {return object;};
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
        Intersections()
        {
            this->intersections = std::priority_queue<Intersection, std::vector<Intersection>, std::greater<Intersection> >();
        }
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
        void push(std::vector<Intersection> intersections)
        {
            for (Intersection i : intersections)
            {
                this->intersections.push(i);
            }
        }
        std::optional<Intersection> top()
        {
            if(!empty()){
                return std::optional<Intersection>(this->intersections.top());
            }
            return std::nullopt;
        }
        bool empty()
        {
            return this->intersections.empty();
        }
        std::optional<Intersection> hit()  
        {
             while (!this->intersections.empty())
             {
                if(this->intersections.top().getTime() >= 0){

                    Intersection result = this->intersections.top();
                    this->intersections.pop();
                    return std::optional<Intersection>(result);
                }
                this->intersections.pop();
             }
             return std::nullopt;
        }
};

class Hitable: public Object
{
    public:
        virtual std::vector<Intersection> intersect(Ray ray) = 0; 
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
            this->material = Material();
            this->material.setAmbient(1);
        }
        Sphere(std::string id, Point position, double radius)
        {
            this->id = id;
            this->position = position;
            this->radius = radius;
        }
        public:
            std::vector<Intersection> intersect(Ray ray)
            {
                Ray localRay = ray.transform(transformation.inverse());
                Tuple object_to_ray = localRay.getOrigin() - this->position;
                double a = localRay.getDirection().dot(localRay.getDirection());
                double b = 2 * localRay.getDirection().dot(object_to_ray);
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
            Vector normalAt(Point world_point){
                Tuple object_point = this->transformation.inverse() * world_point;
                Tuple object_normal = object_point - this->position;
                Vector world_normal = to_vector(this->transformation.inverse().transpose() * object_normal);
                return world_normal.normalize();
            }
};

class Light
{
    protected:
        Point position;
        Color intensity;
    public:
        Point getPosition()
        {
            return this->position;
        }
        Color getIntensity()
        {
            return this->intensity;
        }
};

class PointLight: public Light
{

    public:
        PointLight(Point position, Color intensity)
        {
            this->position = position;
            this->intensity = intensity;
        }

};

class Computation
{
    private:
        double t;
        Object* object;
        Point point;
        Vector eyev;
        Vector normalv;
        bool inside;
        Point over_point;
    public:
        Computation(Intersection intersection, Ray ray)
        {
            this->t = intersection.getTime();
            this->object = intersection.getObject();
            this->point = ray.position(this->t);
            this->eyev = to_vector(-ray.getDirection());
            this->normalv = to_vector(this->object->normalAt(this->point));

            if(this->normalv.dot(this->eyev) < 0)
            {
                this->inside = true;
                this->normalv = to_vector(-this->normalv);
            }
            else
            {
                this->inside = false;
            }

            this->over_point = to_point(this->point + this->normalv * EPSILON);
        }
        Point getPoint() { return this->point; }
        Point getOverPoint() { return this->over_point; }
        Object* getObject() { return this->object; }
        Vector getEyev() { return this->eyev; }
        Vector getNormalv() { return this->normalv; }
        bool isInside() { return this->inside; }
};



Color lighting(Material material, Light light, Point point, Vector eyev, Vector normalv, bool in_shadow)
{

    Color effective_color = to_color(material.getColor() * light.getIntensity());
    Vector lightv = to_vector((light.getPosition() - point).normalize());
    Color ambient = to_color(effective_color * material.getAmbient());
    double light_dot_normal = lightv.dot(normalv);
    Color diffuse;
    Color specular;
    if(light_dot_normal < 0 || in_shadow)
    {
        diffuse = Color::black();
        specular = Color::black();
    }
    else
    {
        diffuse = to_color(effective_color * material.getDiffuse() * light_dot_normal);
        Vector reflectv = to_vector(-lightv).reflect(normalv);
        double reflect_dot_eye = reflectv.dot(eyev);
        if(reflect_dot_eye <= 0)
        {
            specular = Color::black();
        }
        else
        {
            double factor = pow(reflect_dot_eye, material.getShininess());
        }
    }
    return to_color(ambient + diffuse + specular);
}

class World
{
    private:
        Light light;
        std::vector<Object*> objects = std::vector<Object*>();
        bool isShadowed(Point point)
        {
            std::cout << point.toString() << std::endl;
            bool is_shadowed = false;
            Tuple v = this->light.getPosition() - point;
            double distance = v.magnitude();
            Tuple direction = v.normalize();

            Ray r = Ray(point, to_vector(direction));
            Intersections intersections = this->intersectWorld(r);
            
            std::optional<Intersection> h = intersections.hit();
            if(h.has_value() && h.value().getTime() < distance)
            {
                is_shadowed = true;
            }
            return is_shadowed;
        }
        Color shadeHit(Computation comps)
        {
            Point over_point = comps.getOverPoint();
            bool is_shadowed = false; // this->isShadowed(over_point); // TODO: replace with isShadowed function, somehow this is bugged?
            Tuple v = this->light.getPosition() - comps.getOverPoint();
            double distance = v.magnitude();
            Tuple direction = v.normalize();

            Ray r = Ray(comps.getOverPoint(), to_vector(direction));
            Intersections intersections = this->intersectWorld(r);
            
            std::optional<Intersection> h = intersections.hit();
            if(h.has_value() && h.value().getTime() < distance)
            {
                is_shadowed = true;
            }
            return lighting(comps.getObject()->getMaterial(), this->light, over_point, comps.getEyev(), comps.getNormalv(), is_shadowed);
        }
        Intersections intersectWorld(Ray ray)
        {
           Intersections intersections = Intersections();
           for(Object* obj : this->objects)
           {
                Hitable* hitable_obj = dynamic_cast<Hitable*>(obj);
                if(hitable_obj != NULL) {
                    intersections.push(hitable_obj->intersect(ray));
                }
            }
            return intersections;
        }
    public:
        World() {}
        World(std::vector<Object*> objects) { this->objects = objects;}
        static World defaultWorld(std::vector<Object*> objects)
        {
            World default_world = World(objects);
            Light light = PointLight(Point(-10,10,-10), Color(1,1,1));
            default_world.setLight(light);

            return default_world;
        }
        void addObject(Object* object)
        {
            this->objects.push_back(object);
        }
        void setLight(Light light)
        {
            this->light = light;
        }
        Color colorAt(Ray ray)
        {
            Intersections xs = intersectWorld(ray);
            std::optional<Intersection> m_intersection = xs.hit();
            if(m_intersection.has_value()){
                Intersection hit = m_intersection.value();
                Computation comps = Computation(hit, ray);
                return shadeHit(comps);
            }
            return Color::black();
        }
    
};

Matrix view_transform(Point from, Point to, Vector up)
{
    Vector forward = to_vector((to - from).normalize());
    Vector upn = to_vector(up.normalize());
    Vector left = to_vector(forward.cross(upn));
    Vector true_up = left.cross(forward);
    double arr[16] = {left.getX(), left.getY(), left.getZ(), 0,
                      true_up.getX(), true_up.getY(), true_up.getZ(), 0,
                      -forward.getX(), -forward.getY(), -forward.getZ(), 0,
                      0,0,0,1};
    Matrix orientation = Matrix(4,4,arr);
    return orientation.translate(-from.getX(), -from.getY(), -from.getZ());
}

Projectile tick(Environment env, Projectile proj)
{
    Point position = to_point(proj.getPosition() + proj.getVelocity());
    Vector velocity = to_vector(proj.getVelocity() + env.getGravity() + env.getWind());
    return Projectile(position, velocity);
}

class Camera
{
    private:
        int hsize;
        int vsize;
        double field_of_view;
        Matrix transform = Matrix(4,4).identity();
        double pixel_size;
        double half_width;
        double half_height;
       
    public:
        Camera(int hsize, int vsize, double field_of_view) {
            this->hsize = hsize;
            this->vsize = vsize;
            this->field_of_view = field_of_view;

            double half_view = tan(this->field_of_view / 2);
            double aspect = this->hsize / this->vsize;
            if(aspect >= 1)
            {
                this->half_width = half_view;
                this->half_height = half_view / aspect;
            }
            else
            {
                this->half_width = half_view * aspect;
                this->half_height = half_view;
            }
            this->pixel_size =  (this->half_width * 2) / this->hsize;
        }
        Ray rayForPixel(int px, int py)
        {
            double xoffset = (px + 0.5) * this->pixel_size;
            double yoffset = (py + 0.5) * this->pixel_size;

            double world_x = this->half_width - xoffset;
            double world_y = this->half_height - yoffset;

            Tuple pixel = this->transform.inverse() * Point(world_x, world_y, -1);
            Tuple origin = this->transform.inverse() * Point(0,0,0);
            Tuple direction = (pixel - origin).normalize();
            
            return Ray(to_point(origin), to_vector(direction));
        }
        Canvas render(World world)
        {
            Canvas image = Canvas(this->hsize, this->vsize);

            for(int y = 0; y < this->vsize; y ++)
            {
                for(int x = 0; x < this->hsize; x ++)
                {
                    Ray ray = this->rayForPixel(x,y);
                    Color color = world.colorAt(ray);
                    image.writePixel(x,y,color);
                }
            }

            return image;
        }
        void setTransform(Matrix transform)
        {
            this->transform = transform;
        }
        double getPixelSize(){return this->pixel_size;}
};

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
    c.toPPM("chapter1.ppm");
}

void chapter4(){
    Matrix t = Matrix(4,4).identity().translate(0,0,1);
    Tuple origin = Point(0,0,0);
    Canvas c = Canvas(400, 400);
    Tuple twelve = t * origin;
    for(int degree = 0; degree < 360; degree += 30){
        Matrix r = Matrix::translation(200,200,200).scale(150,150,150).rotate_y(radians(degree));
        Tuple p = r * twelve;
        c.writePixel(p.getX(),  p.getZ(), Color(1,1,1));
    }
    c.toPPM("chapter4.ppm");
}

void chapter5(){
    double wallZ = 10;
    double wallSize = 7;
    int canvasPixels = 100;
    double pixelSize = wallSize / canvasPixels;
    double half = wallSize / 2;
    Canvas c = Canvas(canvasPixels,canvasPixels);
    Point rayOrigin = Point(0,0,-5);
    Color red = Color(1,0,0);
    Sphere shape = Sphere();
    for(int y = 0; y < canvasPixels - 1; y ++)
    {
        double world_y = half - pixelSize * y;
        for(int x = 0; x < canvasPixels - 1; x ++)
        {
            double world_x = -half + pixelSize * x;
            Point position = Point(world_x,world_y,wallZ);

            Ray r = Ray(rayOrigin, to_vector((position - rayOrigin).normalize()));
            Intersections xs = Intersections(shape.intersect(r));
            std::optional<Intersection> m_intersection = xs.hit();
            if(m_intersection.has_value()){
                c.writePixel(x,y,red);
            }
        }
    }
    c.toPPM("chapter5.ppm");
}

void chapter6()
{
    Sphere shape = Sphere();
    Material material = Material();
    material.setColor(Color(1, 0.2, 1));
    shape.setMaterial(material);

    Point light_position = Point(-10, 10,-10);
    Color light_color = Color(1,1,1);
    PointLight light = PointLight(light_position, light_color);

    double wallZ = 10;
    double wallSize = 7;
    int canvasPixels = 100;
    double pixelSize = wallSize / canvasPixels;
    double half = wallSize / 2;
    Canvas c = Canvas(canvasPixels,canvasPixels);
    Point rayOrigin = Point(0,0,-5);
    for(int y = 0; y < canvasPixels - 1; y ++)
    {
        double world_y = half - pixelSize * y;
        for(int x = 0; x < canvasPixels - 1; x ++)
        {
            double world_x = -half + pixelSize * x;
            Point position = Point(world_x,world_y,wallZ);

            Ray r = Ray(rayOrigin, to_vector((position - rayOrigin).normalize()));
       
            Intersections xs = Intersections(shape.intersect(r));
            std::optional<Intersection> m_intersection = xs.hit();
            if(m_intersection.has_value()){
                Intersection hit = m_intersection.value();
                Point point = r.position(hit.getTime());
                Vector normal = to_vector(hit.getObject()->normalAt(point));
                Vector eye = to_vector(-r.getDirection());
                Color color = lighting(hit.getObject()->getMaterial(), light, point, eye, normal, false);
                c.writePixel(x,y,color);
            }
        }
    }
    c.toPPM("chapter6.ppm");
}

void chapter6_1()
{
    Sphere shape = Sphere();
    Material material = Material();
    material.setColor(Color(1, 0.2, 1));
    shape.setMaterial(material);
    shape.setTransformation(Matrix::scaling(0.5,0.5,0.5));

    Point light_position = Point(-10, 10,-10);
    Color light_color = Color(1,1,1);
    PointLight light = PointLight(light_position, light_color);

    double wallZ = 10;
    double wallSize = 7;
    int canvasPixels = 100;
    double pixelSize = wallSize / canvasPixels;
    double half = wallSize / 2;
    Canvas c = Canvas(canvasPixels,canvasPixels);
    Point rayOrigin = Point(0,0,-5);
    for(int y = 0; y < canvasPixels - 1; y ++)
    {
        double world_y = half - pixelSize * y;
        for(int x = 0; x < canvasPixels - 1; x ++)
        {
            double world_x = -half + pixelSize * x;
            Point position = Point(world_x,world_y,wallZ);

            Ray r = Ray(rayOrigin, to_vector((position - rayOrigin).normalize()));
       
            Intersections xs = Intersections(shape.intersect(r));
            std::optional<Intersection> m_intersection = xs.hit();
            if(m_intersection.has_value()){
                Intersection hit = m_intersection.value();
                Point point = r.position(hit.getTime());
                Vector normal = to_vector(hit.getObject()->normalAt(point));
                Vector eye = to_vector(-r.getDirection());
                Color color = lighting(hit.getObject()->getMaterial(), light, point, eye, normal, false);
                c.writePixel(x,y,color);
            }
        }
    }
    c.toPPM("chapter6_1.ppm");
}

void chapter6_2()
{
    Sphere shape = Sphere();
    Material material = Material();
    material.setColor(Color(1, 0.2, 1));
    shape.setMaterial(material);
    shape.setTransformation(Matrix::scaling(0.5,1,1) * Matrix::shearing(1,0,0,0,0,0));

    Point light_position = Point(-10, 10,-10);
    Color light_color = Color(1,1,1);
    PointLight light = PointLight(light_position, light_color);

    double wallZ = 10;
    double wallSize = 7;
    int canvasPixels = 100;
    double pixelSize = wallSize / canvasPixels;
    double half = wallSize / 2;
    Canvas c = Canvas(canvasPixels,canvasPixels);
    Point rayOrigin = Point(0,0,-5);
    for(int y = 0; y < canvasPixels - 1; y ++)
    {
        double world_y = half - pixelSize * y;
        for(int x = 0; x < canvasPixels - 1; x ++)
        {
            double world_x = -half + pixelSize * x;
            Point position = Point(world_x,world_y,wallZ);

            Ray r = Ray(rayOrigin, to_vector((position - rayOrigin).normalize()));
       
            Intersections xs = Intersections(shape.intersect(r));
            std::optional<Intersection> m_intersection = xs.hit();
            if(m_intersection.has_value()){
                Intersection hit = m_intersection.value();
                Point point = r.position(hit.getTime());
                Vector normal = to_vector(hit.getObject()->normalAt(point));
                Vector eye = to_vector(-r.getDirection());
                Color color = lighting(hit.getObject()->getMaterial(), light, point, eye, normal, false);
                c.writePixel(x,y,color);
            }
        }
    }
    c.toPPM("chapter6_2.ppm");
}

void chapter6_3()
{
    Sphere shape = Sphere();
    Material material = Material();
    material.setColor(Color(1, 0.2, 1));
    shape.setMaterial(material);
    shape.setTransformation(Matrix::rotation_z(radians(90)).scale(1,0.5,0.5));

    Point light_position = Point(-10, 10,-10);
    Color light_color = Color(1,1,1);
    PointLight light = PointLight(light_position, light_color);

    double wallZ = 10;
    double wallSize = 7;
    int canvasPixels = 100;
    double pixelSize = wallSize / canvasPixels;
    double half = wallSize / 2;
    Canvas c = Canvas(canvasPixels,canvasPixels);
    Point rayOrigin = Point(0,0,-5);
    for(int y = 0; y < canvasPixels - 1; y ++)
    {
        double world_y = half - pixelSize * y;
        for(int x = 0; x < canvasPixels - 1; x ++)
        {
            double world_x = -half + pixelSize * x;
            Point position = Point(world_x,world_y,wallZ);

            Ray r = Ray(rayOrigin, to_vector((position - rayOrigin).normalize()));
       
            Intersections xs = Intersections(shape.intersect(r));
            std::optional<Intersection> m_intersection = xs.hit();
            if(m_intersection.has_value()){
                Intersection hit = m_intersection.value();
                Point point = r.position(hit.getTime());
                Vector normal = to_vector(hit.getObject()->normalAt(point));
                Vector eye = to_vector(-r.getDirection());
                Color color = lighting(hit.getObject()->getMaterial(), light, point, eye, normal, false);
                c.writePixel(x,y,color);
            }
        }
    }
    c.toPPM("chapter6_3.ppm");
}

void chapter7()
{
    Sphere floor = Sphere();
    floor.setTransformation(Matrix::scaling(10, 0.01, 10));
    Material floor_material = Material();
    floor_material.setColor(Color(1,0.9,0.9));
    floor_material.setSpecular(0);
    floor.setMaterial(floor_material);

    Sphere left_wall = Sphere();
    left_wall.setTransformation(Matrix::translation(0,0,5) * Matrix::rotation_y(-M_PI/4) * Matrix::rotation_x(M_PI/2) * Matrix::scaling(10,0.01,10));
    left_wall.setMaterial(Material());

    Sphere right_wall = Sphere();
    right_wall.setTransformation(Matrix::translation(0,0,5) * Matrix::rotation_y(M_PI/4) * Matrix::rotation_x(M_PI/2) * Matrix::scaling(10,0.01,10));
    right_wall.setMaterial(Material());

    Sphere middle = Sphere();
    Material middle_material = Material();
    middle_material.setColor(Color(0.1,1,0.5));
    middle_material.setDiffuse(0.7);
    middle_material.setSpecular(0.3);
    middle.setMaterial(middle_material);
    middle.setTransformation(Matrix::translation(-0.5,1,0.5));

    Sphere right = Sphere();
    Material right_material = Material();
    right_material.setColor(Color(0.5,1,0.1));
    right_material.setDiffuse(0.7);
    right_material.setSpecular(0.3);
    right.setMaterial(right_material);
    right.setTransformation(Matrix::translation(1.5,0.5,-0.5) * Matrix::scaling(0.5,0.5,0.5));

    Sphere left = Sphere();
    Material left_material = Material();
    left_material.setColor(Color(1,0.8,0.1));
    left_material.setDiffuse(0.7);
    left_material.setSpecular(0.3);
    left.setMaterial(left_material);
    left.setTransformation(Matrix::translation(-1.5,0.33,-0.75) * Matrix::scaling(0.33,0.33,0.33));

    World world = World();
    world.setLight(PointLight(Point(-10,10,-10), Color(1,1,1)));
    world.addObject(&middle);
    world.addObject(&left);
    world.addObject(&right);
    world.addObject(&left_wall);
    world.addObject(&right_wall);
    world.addObject(&floor);
  
    Camera camera = Camera(100,50,M_PI /3);
    camera.setTransform(view_transform(Point(0,1.5,-5),Point(0,1,0),Vector(0,1,0)));

    Canvas canvas = camera.render(world);
    canvas.toPPM("chapter7.ppm");
}

void multiplying_a_product_by_its_inverse()
{
    double arr1[16] = {3,-9, 7,3,
                       3,-8, 2,-9,
                      -4,4,4,1,
                      -6,5,-1,1 };
    double arr2[16] = {8,2,2,2,
                       3,-1,7,0,
                       7,0,5,4,
                       6,-2,0,5};
    Matrix a = Matrix(4,4,arr1);
    Matrix b = Matrix(4,4,arr2);
    Matrix c = a * b;
    Matrix _a = c * b.inverse();
    assert(a.toString() == _a.toString() && "c = a * b == c * inverse(b) = a");
}
// void shadow_is_hit(){
//     Sphere s1 = Sphere();
//     Material m1 = Material();
//     m1.setColor(Color(0.8,0.1,0.6));
//     m1.setDiffuse(0.7);
//     m1.setSpecular(0.2);
//     s1.setMaterial(m1);
//     Sphere s2 = Sphere();
//     s2.setTransformation(Matrix::scaling(0.5,0.5,0.5));

//     std::vector<Object*> objects = std::vector<Object*>();
//     objects.push_back(&s1);
//     objects.push_back(&s2);
//     World w = World::defaultWorld(objects);
//     bool b = w.isShadowed(Point(0,10,0));
//     assert(b == false && "No shadows");
//     bool b2 = w.isShadowed(Point(10,-10,10));
//     assert(b2 == true && "Shadows");
// }
void feature_matrices()
{
    multiplying_a_product_by_its_inverse();
}

// void feature_shadows()
// {
//     shadow_is_hit();
// }

int main(int argc, char * argv[])
{
    feature_matrices();
    // feature_shadows();

    chapter1();
    chapter4();
    chapter5();
    chapter6();
    chapter6_1();
    chapter6_2();
    chapter6_3();
    chapter7();

    return 0;
}

