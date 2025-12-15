#include <format>
#include <iostream>

#include <Eigen/Dense>

struct CPose3DQuat
{
    double tx, ty, tz;
    Eigen::Quaterniond q;

    Eigen::Quaterniond quat() const { return q; }
    double x() const { return tx; }
    double y() const { return ty; }
    double z() const { return tz; }
};

typedef Eigen::Matrix<double, 7, 7> CMatrixDouble77;
typedef Eigen::Matrix<double, 4, 4> CMatrixDouble44;

CPose3DQuat generateRandomPose(void)
{
    CPose3DQuat pose;
    pose.tx = Eigen::internal::random<double>(-10.0, 10.0);
    pose.ty = Eigen::internal::random<double>(-10.0, 10.0);
    pose.tz = Eigen::internal::random<double>(-10.0, 10.0);
    pose.q = Eigen::Quaterniond::UnitRandom();
    return pose;
}

CMatrixDouble77 generateRandomCovarianceMatrix(void)
{
    CMatrixDouble77 A = CMatrixDouble77::Zero();
    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 7; j++)
        {
            A(i, j) = Eigen::internal::random<double>(-1.0, 1.0);
        }
    }

    return A.transpose() * A + Eigen::Matrix<double, 7, 7>::Identity() * 1e-9;
}

double square(double x)
{
    return x * x;
}

CMatrixDouble44 normalizationJacobian(const Eigen::Quaterniond& q)
{
    const double w = q.w();
    const double x = q.x();
    const double y = q.y();
    const double z = q.z();

    const double s = q.squaredNorm();

    const double n = 1.0 / std::pow(s, 1.5);

    CMatrixDouble44 J;
    J(0,0) =  x*x + y*y + z*z;  J(0,1) = -w*x;             J(0,2) = -w*y;            J(0,3) = -w*z;
    J(1,0) = -x*w;              J(1,1) =  w*w + y*y + z*z; J(1,2) = -x*y;            J(1,3) = -x*z;
    J(2,0) = -y*w;              J(2,1) = -y*x;             J(2,2) = w*w + x*x + z*z; J(2,3) = -y*z;
    J(3,0) = -z*w;              J(3,1) = -z*x;             J(3,2) = -z*y;            J(3,3) =  w*w + x*x + y*y;

    J *= n;
    return J;
}

void jacobiansPoseComposition(
    const CPose3DQuat& x,
    const CPose3DQuat& u,
    CMatrixDouble77& df_dx,
    CMatrixDouble77& df_du,
    bool normalise = true
)
{
    const double qr = x.quat().w();
    const double qx = x.quat().x();
    const double qx2 = square(qx);
    const double qy = x.quat().y();
    const double qy2 = square(qy);
    const double qz = x.quat().z();
    const double qz2 = square(qz);

    const double ax = u.x();
    const double ay = u.y();
    const double az = u.z();
    const double q2r = u.quat().w();
    const double q2x = u.quat().x();
    const double q2y = u.quat().y();
    const double q2z = u.quat().z();

    Eigen::Quaterniond x_plus_u_q = x.quat() * u.quat(); // Note: Unlike MRPT, Eigen does not normalise automatically
    if (normalise)
    {
        x_plus_u_q.normalize();
    }
    
    const CMatrixDouble44 norm_jacob   = normalizationJacobian(x_plus_u_q);
    const CMatrixDouble44 norm_jacob_x = normalizationJacobian(x.q);

    df_dx.setZero();
    df_dx(0, 0) = 1.0;
    df_dx(1, 1) = 1.0;
    df_dx(2, 2) = 1.0;

    Eigen::Matrix<double, 3, 4> vals2;
    vals2 <<
        2.0 * (-qz * ay + qy * az),
        2.0 * ( qy * ay + qz * az),
        2.0 * (-2.0 * qy * ax + qx * ay + qr * az),
        2.0 * (-2.0 * qz * ax - qr * ay + qx * az),

        2.0 * ( qz * ax - qx * az),
        2.0 * ( qy * ax - 2.0 * qx * ay - qr * az),
        2.0 * ( qx * ax + qz * az),
        2.0 * ( qr * ax - 2.0 * qz * ay + qy * az),

        2.0 * (-qy * ax + qx * ay),
        2.0 * ( qz * ax + qr * ay - 2.0 * qx * az),
        2.0 * (-qr * ax + qz * ay - 2.0 * qy * az),
        2.0 * ( qx * ax + qy * ay);

    df_dx.block<3,4>(0,3) = vals2 * norm_jacob_x;

    Eigen::Matrix<double, 4, 4> aux44_x;
    aux44_x <<
        q2r, -q2x, -q2y, -q2z,
        q2x,  q2r,  q2z, -q2y,
        q2y, -q2z,  q2r,  q2x,
        q2z,  q2y, -q2x,  q2r;

    df_dx.block<4,4>(3,3) = norm_jacob * aux44_x;

    df_du.setZero();
    df_du(0, 0) = 1.0 - 2.0 * (qy2 + qz2);
    df_du(0, 1) = 2.0 * (qx * qy - qr * qz);
    df_du(0, 2) = 2.0 * (qr * qy + qx * qz);

    df_du(1, 0) = 2.0 * (qr * qz + qx * qy);
    df_du(1, 1) = 1.0 - 2.0 * (qx2 + qz2);
    df_du(1, 2) = 2.0 * (qy * qz - qr * qx);

    df_du(2, 0) = 2.0 * (qx * qz - qr * qy);
    df_du(2, 1) = 2.0 * (qr * qx + qy * qz);
    df_du(2, 2) = 1.0 - 2.0 * (qx2 + qy2);

    Eigen::Matrix<double, 4, 4> aux44_u;
    aux44_u <<
        qr, -qx, -qy, -qz,
        qx,  qr, -qz,  qy,
        qy,  qz,  qr, -qx,
        qz, -qy,  qx,  qr;

    df_du.block<4,4>(3,3) = norm_jacob * aux44_u;

    return;
}


int main(int argc, char const *argv[])
{
    // tests
    return 0;
}

