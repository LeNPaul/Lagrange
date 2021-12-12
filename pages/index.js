import { Container, Box, Heading } from '@chakra-ui/react'

const Page = () => {
    return (
        <Container>
            <Box borderRadius="lg" bg="red" p={3} mb={6} align="center">
                Hello, I&apos;m a computer science student at the University of Minnesota!
            </Box>

            <Box display={{md:'flex'}}>
                <Box flexGrow={1}>
                    <Heading as="h2" variant="page-title">
                        John Anselmo
                    </Heading>
                    <p>Student ( Developer / Photographer / Runner )</p>
                </Box>
            </Box>
        </Container>
    )
}

export default Page
