using TPSChem
using LinearAlgebra
using Test

@testset "absorption_spectrum" begin

    @testset "compute_transition_dipoles" begin
        norb = 3
        R = 4
        γ_aa = zeros(norb, norb, R, R)
        γ_bb = zeros(norb, norb, R, R)
        # diagonal: each root has equal α/β population at orbital 1
        for r in 1:R
            γ_aa[1, 1, r, r] = 0.5
            γ_bb[1, 1, r, r] = 0.5
        end
        # off-diagonal: root 1→2 transition only in x via orbital 1
        γ_aa[1, 1, 1, 2] = 1.0
        γ_bb[1, 1, 1, 2] = 0.5

        dip_x = zeros(norb, norb); dip_x[1, 1] = 1.0
        dip_y = zeros(norb, norb); dip_y[2, 2] = 1.0
        dip_z = zeros(norb, norb)

        tdm_x, tdm_y, tdm_z = compute_transition_dipoles(γ_aa, γ_bb, dip_x, dip_y, dip_z)

        @test length(tdm_x) == R
        # permanent dipole of root 1: sum_{pq} dip_x[p,q] * (γ_aa[p,q,1,1] + γ_bb[p,q,1,1]) = 1.0
        @test isapprox(tdm_x[1], 1.0, atol=1e-14)
        # root 1→2: γ_aa[1,1,1,2]=1.0 + γ_bb[1,1,1,2]=0.5 → tdm_x[2] = 1.5
        @test isapprox(tdm_x[2], 1.5, atol=1e-14)
        # no transition along y for root 1→2 (γ[2,2,1,2] = 0)
        @test isapprox(tdm_y[2], 0.0, atol=1e-14)
        # z always zero
        @test all(isapprox.(tdm_z, 0.0, atol=1e-14))

        # mismatched R1 != R2 should throw
        γ_bad = zeros(norb, norb, R, R+1)
        @test_throws ArgumentError compute_transition_dipoles(γ_bad, γ_bad, dip_x, dip_y, dip_z)
    end

    @testset "compute_oscillator_strengths" begin
        norb = 2
        R = 3
        E = Float64[0.0, 1.0, 2.0]
        γ_aa = zeros(norb, norb, R, R)
        γ_bb = zeros(norb, norb, R, R)
        # transition 1→2 along x only
        γ_aa[1, 1, 1, 2] = 1.0
        dip_x = [1.0 0.0; 0.0 0.0]
        dip_y = zeros(norb, norb)
        dip_z = zeros(norb, norb)

        f = compute_oscillator_strengths(E, γ_aa, γ_bb, dip_x, dip_y, dip_z)

        @test length(f) == R
        @test f[1] == 0.0   # ref root is always zero by convention
        # f[2] = (2/3) * ΔE * |tdm_x[2]|^2 = (2/3)*1.0*1.0^2
        @test isapprox(f[2], 2.0/3.0, atol=1e-14)
        # root 3: no off-diagonal γ set → f[3] = 0
        @test f[3] == 0.0

        # wrong E length should throw
        @test_throws DimensionMismatch compute_oscillator_strengths(E[1:2], γ_aa, γ_bb, dip_x, dip_y, dip_z)
    end

    @testset "absorption_spectrum lorentzian" begin
        E = Float64[0.0, 0.5, 1.5]
        f = Float64[0.0, 1.0, 0.3]
        ω, I = absorption_spectrum(E, f; σ=0.05, npoints=300, lineshape=:lorentzian)

        @test length(ω) == 300
        @test length(I) == 300
        @test all(I .>= 0.0)
        # dominant peak must be near ω=0.5 (f=1.0, the stronger transition)
        @test isapprox(ω[argmax(I)], 0.5, atol=0.1)
    end

    @testset "absorption_spectrum gaussian" begin
        E = Float64[0.0, 1.0]
        f = Float64[0.0, 1.0]
        ω, I = absorption_spectrum(E, f; σ=0.02, npoints=500, lineshape=:gaussian)

        @test length(ω) == 500
        @test all(I .>= 0.0)
        @test isapprox(ω[argmax(I)], 1.0, atol=0.05)
        # integral of a normalised Gaussian is 1 → ≈ f[2] = 1.0
        Δω = ω[2] - ω[1]
        @test isapprox(sum(I) * Δω, 1.0, atol=0.02)
    end

    @testset "absorption_spectrum custom ω range" begin
        E = Float64[0.0, 1.0]
        f = Float64[0.0, 1.0]
        ω, I = absorption_spectrum(E, f; ω_min=0.8, ω_max=1.2, npoints=100)

        @test length(ω) == 100
        @test isapprox(ω[1],   Float64(0.8), atol=1e-12)
        @test isapprox(ω[end], Float64(1.2), atol=1e-12)
    end

    @testset "absorption_spectrum no transitions above thresh" begin
        E = Float64[0.0, 1.0, 2.0]
        f = Float64[0.0, 1e-12, 0.0]   # all below default thresh=1e-10
        ω, I = absorption_spectrum(E, f; thresh=1e-10)

        @test isempty(ω)
        @test isempty(I)
    end

    @testset "absorption_spectrum unknown lineshape" begin
        E = Float64[0.0, 1.0]
        f = Float64[0.0, 1.0]
        @test_throws ErrorException absorption_spectrum(E, f; lineshape=:square)
    end

    @testset "print_stick_spectrum" begin
        E = Float64[0.0, 0.4, 1.2]
        f = Float64[0.0, 0.3, 0.1]
        @test_nowarn print_stick_spectrum(E, f)
        @test_nowarn print_stick_spectrum(E, f; units=:ev)
    end
end
