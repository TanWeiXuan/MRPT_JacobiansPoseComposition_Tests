#include <array>
#include <cassert>
#include <format>
#include <iostream>
#include <vector>

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

double square(double x)
{
    return x * x;
}

template <int N>
Eigen::Matrix<double, N, N> generateRandomSpdMatrix()
{
    Eigen::Matrix<double, N, N> A = Eigen::Matrix<double, N, N>::Zero();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A(i, j) = Eigen::internal::random<double>(-0.001, 0.001);
        }
    }

    return A.transpose() * A + Eigen::Matrix<double, N, N>::Identity() * 1e-9;
}

CMatrixDouble77 generateRandomCovarianceMatrix(const Eigen::Quaterniond& q)
{
    CMatrixDouble77 P = CMatrixDouble77::Zero();

    // Translation covariance (positive definite)
    P.block<3,3>(0,0) = generateRandomSpdMatrix<3>();

    // Build an orthonormal basis for the tangent space at q (vectors orthogonal to q)
    const Eigen::Vector4d q_vec(q.w(), q.x(), q.y(), q.z());
    const Eigen::Vector4d q_unit = q_vec.normalized();

    std::vector<Eigen::Vector4d> basis;
    basis.reserve(3);

    const std::array<Eigen::Vector4d, 4> canonical_basis = {
        Eigen::Vector4d::Unit(0),
        Eigen::Vector4d::Unit(1),
        Eigen::Vector4d::Unit(2),
        Eigen::Vector4d::Unit(3)
    };

    for (const auto& e : canonical_basis)
    {
        Eigen::Vector4d v = e - (q_unit.dot(e)) * q_unit;
        for (const auto& b : basis)
        {
            v -= (b.dot(v)) * b;
        }

        const double norm_v = v.norm();
        if (norm_v > 1e-12)
        {
            basis.push_back(v / norm_v);
        }

        if (basis.size() == 3) break;
    }

    Eigen::Matrix<double, 4, 3> B = Eigen::Matrix<double, 4, 3>::Zero();
    for (size_t i = 0; i < basis.size(); ++i)
    {
        B.col(static_cast<int>(i)) = basis[i];
    }

    assert(basis.size() == 3 && "Quaternion tangent basis should have rank 3");

    // Quaternion covariance constrained to the tangent space:
    //  - rank <= 3
    //  - Cq = 0
    const Eigen::Matrix<double, 3, 3> quat_cov_tangent = generateRandomSpdMatrix<3>();
    const CMatrixDouble44 quat_cov = B * quat_cov_tangent * B.transpose();

    P.block<4,4>(3,3) = quat_cov;

    return P;
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
    constexpr int NUM_TRIALS = 5000;
    constexpr int NUM_SAMPLES_TO_PRINT = 5; 
    constexpr double QUATERNION_ERROR = 1e-6; 

    double sum_P = 0.0, sum_rel_P = 0.0, max_P = 0.0, max_rel_P = 0.0;

    for (int k = 0; k < NUM_TRIALS; ++k)
    {
        CPose3DQuat x = generateRandomPose();
        CPose3DQuat u = generateRandomPose();

        // Add some small amount of noise to the quaternions so that their norms are not exactly 1
        x.q.coeffs()(0) += Eigen::internal::random<double>(-QUATERNION_ERROR, QUATERNION_ERROR);
        x.q.coeffs()(1) += Eigen::internal::random<double>(-QUATERNION_ERROR, QUATERNION_ERROR);
        x.q.coeffs()(2) += Eigen::internal::random<double>(-QUATERNION_ERROR, QUATERNION_ERROR);
        x.q.coeffs()(3) += Eigen::internal::random<double>(-QUATERNION_ERROR, QUATERNION_ERROR);

        u.q.coeffs()(0) += Eigen::internal::random<double>(-QUATERNION_ERROR, QUATERNION_ERROR);
        u.q.coeffs()(1) += Eigen::internal::random<double>(-QUATERNION_ERROR, QUATERNION_ERROR);
        u.q.coeffs()(2) += Eigen::internal::random<double>(-QUATERNION_ERROR, QUATERNION_ERROR);
        u.q.coeffs()(3) += Eigen::internal::random<double>(-QUATERNION_ERROR, QUATERNION_ERROR);

        CMatrixDouble77 dx_norm, du_norm; // with normalization
        CMatrixDouble77 dx_no_norm, du_no_norm; // without normalization

        jacobiansPoseComposition(x, u, dx_norm, du_norm, true);
        jacobiansPoseComposition(x, u, dx_no_norm,  du_no_norm,  false);

        // Compare propagated differences:
        const CMatrixDouble77 Px = generateRandomCovarianceMatrix(x.q);
        const CMatrixDouble77 Pu = generateRandomCovarianceMatrix(u.q);

        const CMatrixDouble77 P_norm = dx_norm * Px * dx_norm.transpose()
                                     + du_norm * Pu * du_norm.transpose();

        const CMatrixDouble77 P_no_norm  = dx_no_norm  * Px * dx_no_norm.transpose()
                                         + du_no_norm  * Pu * du_no_norm.transpose();

        const double nP = P_no_norm.block<4,4>(3,3).norm();
        const double diff_P = (P_norm.block<4,4>(3,3) - P_no_norm.block<4,4>(3,3)).norm();
        const double rel_P  = diff_P / std::max(nP, 1e-30);

        sum_P += diff_P;
        sum_rel_P += rel_P;
        max_P = std::max(max_P, diff_P);
        max_rel_P = std::max(max_rel_P, rel_P);

        // Print first few samples:
        if (k < NUM_SAMPLES_TO_PRINT)
        {
            std::cout << "----------------------------------------\n";
            std::cout << std::format("Sample {}:\n", k+1);
            std::cout << "With normalization:\n" << P_norm.block<4,4>(3,3) << "\n";
            std::cout << "Without normalization:\n" << P_no_norm.block<4,4>(3,3) << "\n";
            std::cout << "Difference:\n" << P_norm.block<4,4>(3,3) - P_no_norm.block<4,4>(3,3) << "\n";
            std::cout << std::format("Difference in P norm: {:.6e}, relative: {:.6e}\n", diff_P, rel_P);
            std::cout << std::format("x.quat().norm() - 1.0: {:.6e}\n", x.q.norm() - 1.0);
            std::cout << std::format("u.quat().norm() - 1.0: {:.6e}\n", u.q.norm() - 1.0);
        }
    }

    std::cout << "========================================\n";
    std::cout << std::format("Average difference in P norm: {:.6e}, relative: {:.6e}\n", sum_P / NUM_TRIALS, sum_rel_P / NUM_TRIALS);
    std::cout << std::format("Maximum difference in P norm: {:.6e}, relative: {:.6e}\n", max_P, max_rel_P);

    return 0;
}
